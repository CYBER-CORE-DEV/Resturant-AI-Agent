"""
Restaurant ordering bot, built with LangGraph and Groq (openai/gpt-oss-120b).

Every order is one run of the graph:

    take_order -> cook -> serve -> payment -> END

If something goes wrong on the way (dish not available, oil finished, nobody free
to serve...) the graph jumps straight to END and the customer gets a message
saying what happened.

Setup:
    pip install langgraph groq python-dotenv qrcode
    put GROQ_API_KEY=... in your .env file
    (qrcode is optional, without it the payment step prints the code as plain text)
"""

import io
import itertools
import json
import os
import random
from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, START, StateGraph

try:
    import qrcode
except ImportError:
    qrcode = None

load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

# price in rupees and how many portions are left
MENU = {
    "margherita pizza": {"price": 250, "quantity": 5},
    "veg burger": {"price": 120, "quantity": 10},
    "paneer tikka": {"price": 220, "quantity": 4},
    "pasta": {"price": 180, "quantity": 6},
    "french fries": {"price": 90, "quantity": 8},
    "chocolate shake": {"price": 150, "quantity": 0},  # sold out, handy for testing
}

KITCHEN = {"oil": 3}      # every order uses 1 oil, at 0 we can't cook anymore
SERVE_FAILURE_RATE = 0.2  # 0 = serving never fails, 1 = always fails
DARK_TERMINAL = True      # set to False if the QR looks inverted on a light terminal

order_numbers = itertools.count(1001)


# tools: plain python functions. take_order is the only one the LLM calls itself,
# cook, serve and payment are run by the graph so the steps always happen in order

def take_order(item, quantity):
    name = str(item).strip().lower()
    label = name.title() or "that dish"

    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        quantity = 0
    if quantity < 1:
        return {"ok": False, "message": "Sorry, I didn't catch how many you want. How many would you like?"}

    dish = MENU.get(name)
    if dish is None or dish["quantity"] == 0:
        return {"ok": False, "message": f"Sorry, {label} is not available right now."}

    left = dish["quantity"]
    if quantity > left:
        return {
            "ok": False,
            "message": (f"Sorry, we only have {left} {label} left but you asked for {quantity}. "
                        "Please order less or pick something else."),
        }

    dish["quantity"] -= quantity  # keep these portions aside for this order
    return {"ok": True, "item": name, "quantity": quantity, "total_price": dish["price"] * quantity}


def release_order(item, quantity):
    # cooking failed, so the portions go back on the shelf
    MENU[item]["quantity"] += quantity


def cook(item, quantity):
    if KITCHEN["oil"] < 1:
        return {"ok": False, "reason": "the kitchen has run out of cooking oil"}
    KITCHEN["oil"] -= 1
    return {"ok": True}


def serve(item, quantity):
    if random.random() < SERVE_FAILURE_RATE:
        return {"ok": False, "reason": "no waiter is free to bring it to your table"}
    return {"ok": True}


def payment_code(order_id, amount):
    # just text for the demo, it is not a real payment link
    return f"DEMO-PAY|{order_id}|INR {amount}|NOT A REAL PAYMENT"


def draw_qr(text):
    if qrcode is None:
        return f"(pip install qrcode to see a real QR here)\nCode contents: {text}"

    qr = qrcode.QRCode()
    qr.add_data(text)
    qr.make(fit=True)
    out = io.StringIO()
    qr.print_ascii(out=out, invert=DARK_TERMINAL)
    return out.getvalue()


# what we tell the LLM about the take_order tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "take_order",
            "description": (
                "Place an order for ONE dish. Checks the menu and the stock and keeps the "
                "portions aside if they are available. Call it whenever the customer asks "
                "for a dish, even if the dish might not be on the menu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "Dish name as written on the menu, e.g. 'Margherita Pizza'.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "How many portions, at least 1. Use 1 for 'a' or 'one'.",
                    },
                },
                "required": ["item", "quantity"],
            },
        },
    }
]

menu_text = ", ".join(f"{name.title()} (Rs {dish['price']})" for name, dish in MENU.items())

SYSTEM_PROMPT = f"""You are the order taker at a small restaurant. You only help people order food.

Menu: {menu_text}.

Rules:
- When someone wants a dish, call take_order with the dish and how many. Do this even if the
  dish is not on the menu, the tool checks what is really available.
- If you can't tell which dish or how many, ask one short question.
- One dish per order. If they ask for several, order the first one and tell them to order
  the others afterwards.
- If the message is not about ordering food, politely say you can only help with food orders.
  Don't answer the question. You can tell them what is on the menu.
- Never say an order has been placed yourself, only the tool places orders."""


class State(TypedDict):
    user_input: str   # what the customer just typed
    history: list     # earlier messages, so follow-ups like "make it 3" work
    item: str
    quantity: int
    total_price: int
    order_id: str
    status: str       # the routers below only look at this
    message: str      # what the customer sees at the end
    payment_qr: str   # the QR drawn as text, only filled in when it's time to pay

    # status goes: new -> no_order | order_failed | ordered
    #              ordered -> cook_failed | cooked
    #              cooked -> serve_failed | served
    #              served -> awaiting_payment


def step(node, **update):
    # prints which node just ran and hands the update back to langgraph
    print(f"  [{node}] status = {update['status']}")
    return update


# nodes

def take_order_node(state: State):
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + state["history"]
        + [{"role": "user", "content": state["user_input"]}]
    )

    try:
        response = groq.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice="auto"
        )
    except Exception as e:
        return step("take_order", status="no_order",
                    message=f"Sorry, something went wrong on our side ({e}). Please try again.")

    reply = response.choices[0].message

    # no tool call: not about food, or the LLM needs to ask something first
    if not reply.tool_calls:
        return step("take_order", status="no_order",
                    message=reply.content or "Sorry, I can only help with food orders.")

    # we only handle the first dish
    try:
        args = json.loads(reply.tool_calls[0].function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    result = take_order(args.get("item", ""), args.get("quantity", 1))

    if not result["ok"]:
        return step("take_order", status="order_failed", message=result["message"])

    return step("take_order", status="ordered",
                item=result["item"], quantity=result["quantity"],
                total_price=result["total_price"], order_id=f"ORD-{next(order_numbers)}")


def cook_node(state: State):
    result = cook(state["item"], state["quantity"])

    if not result["ok"]:
        release_order(state["item"], state["quantity"])
        dish = f"{state['quantity']} x {state['item'].title()}"
        return step("cook", status="cook_failed",
                    message=(f"Sorry, we couldn't cook your {dish}: {result['reason']}. "
                             "Your order has been cancelled, please place a new order."))

    return step("cook", status="cooked")


def serve_node(state: State):
    result = serve(state["item"], state["quantity"])
    dish = f"{state['quantity']} x {state['item'].title()}"

    if not result["ok"]:
        return step("serve", status="serve_failed",
                    message=(f"Your {dish} was cooked but we couldn't serve it: {result['reason']}. "
                             "Please call a staff member."))

    return step("serve", status="served",
                message=f"Here is your {dish}. Total: Rs {state['total_price']}. Enjoy your meal!")


def payment_node(state: State):
    code = payment_code(state["order_id"], state["total_price"])
    message = (f"{state['message']}\n\n"
               f"Order ID: {state['order_id']}. Please scan the QR code below to pay "
               f"Rs {state['total_price']} (DEMO code, no real payment is made).")
    return step("payment", status="awaiting_payment", message=message, payment_qr=draw_qr(code))


# edges: decide where to go next from the status

def after_take_order(state: State):
    return "cook" if state["status"] == "ordered" else "end"


def after_cook(state: State):
    return "serve" if state["status"] == "cooked" else "end"


def after_serve(state: State):
    return "payment" if state["status"] == "served" else "end"


builder = StateGraph(State)
builder.add_node("take_order", take_order_node)
builder.add_node("cook", cook_node)
builder.add_node("serve", serve_node)
builder.add_node("payment", payment_node)

builder.add_edge(START, "take_order")
builder.add_conditional_edges("take_order", after_take_order, {"cook": "cook", "end": END})
builder.add_conditional_edges("cook", after_cook, {"serve": "serve", "end": END})
builder.add_conditional_edges("serve", after_serve, {"payment": "payment", "end": END})
builder.add_edge("payment", END)

graph = builder.compile()


def run_order(user_input, history):
    return graph.invoke({
        "user_input": user_input,
        "history": history,
        "item": "",
        "quantity": 0,
        "total_price": 0,
        "order_id": "",
        "status": "new",
        "message": "",
        "payment_qr": "",
    })


def main():
    print("Welcome to our restaurant! What would you like to order? (type 'quit' to leave)")
    history = []

    # each trip through this loop is one run of the graph, so after a failure the
    # customer just types a new order
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue

        result = run_order(text, history)
        print(f"\nBot: {result['message']}")
        if result["payment_qr"]:
            print(result["payment_qr"])

        # keep the last few messages, but not the QR, it would only waste tokens
        history += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": result["message"]},
        ]
        history = history[-10:]


if __name__ == "__main__":
    main()