import json
import argparse
from pathlib import Path
from mlx_lm import load, generate


parser = argparse.ArgumentParser()
parser.add_argument("--messages", required=True)
args = parser.parse_args()

messages_input = json.loads(args.messages)
system, *conversation = messages_input

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model_llama_finetuning" / "llama_ventas_colombiano_mlx_q8"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
SYSTEM_PROMPT = (
    "Eres un vendedor colombiano amigable y cercano. "
    "Hablas con palabras y expresiones típicas de Colombia.\n\n"

    "--- USO DE HERRAMIENTAS ---\n"

    "1. Si el cliente pregunta por productos disponibles usa 'get_products'.\n"
    "Responde SOLO con este JSON:\n"
    "{\"tool\": \"get_products\", \"query\": \"término\"}\n\n"

    "2. Si el cliente confirma la compra usa 'create_order'.\n"
    "Responde SOLO con este JSON:\n"
    "{\"tool\": \"create_order\", \"product_name\": \"nombre\", "
    "\"quantity\": cantidad, \"price\": precio, \"customer_phone\": telefono}\n\n"

    "3. Si el cliente solo está saludando o conversando, responde normalmente "
    "como vendedor colombiano y NO uses herramientas.\n"
)

model, tokenizer = load(MODEL_PATH)

def get_internal_knowledge(query, top_k=3):
    return f"Resultados de productos para: {query}"


def create_order(product_name, quantity, price, customer_phone):
    return f"Orden creada: {quantity} x {product_name} por {price} para {customer_phone}"


def generate_response_from_model(
    model,
    tokenizer,
    max_tokens=DEFAULT_MAX_TOKENS,
    top_k=3,
):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *conversation
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = generate(
        model,
        tokenizer,
        prompt=chat_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )

    text = response.strip()

    try:
        tool_call = json.loads(text)

        if isinstance(tool_call, dict) and "tool" in tool_call:

            tool_name = tool_call["tool"]
            if tool_name == "get_products":

                query_tool = tool_call.get("query", "")
                knowledge = get_internal_knowledge(query_tool, top_k=top_k)
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "system",
                        "content": f"RESULTADO DE LA HERRAMIENTA:\n{knowledge}",
                    }
                )

            elif tool_name == "create_order":

                order_status = create_order(
                    product_name=tool_call.get("product_name"),
                    quantity=tool_call.get("quantity"),
                    price=tool_call.get("price"),
                    customer_phone=tool_call.get("customer_phone"),
                )

                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "system",
                        "content": f"RESULTADO DE LA HERRAMIENTA:\n{order_status}",
                    }
                )

            chat_prompt_final = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            final_response = generate(
                model,
                tokenizer,
                prompt=chat_prompt_final,
                max_tokens=max_tokens,
                verbose=False,
            )

            text = final_response.strip()

    except json.JSONDecodeError:
        pass

    print(f"RESULT_TEXT={text}")


if __name__ == "__main__":
    generate_response_from_model(
        model,
        tokenizer
    )