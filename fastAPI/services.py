from typing import Dict, List, Optional, Any
from mlx_lm import load, generate
from config import MODEL_PATH, DEFAULT_MAX_TOKENS, SYSTEM_PROMPT, embed_model, collection
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load(str(MODEL_PATH))

def _build_messages(
    user_prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    knowledge: Optional[str] = None,
) -> List[Dict[str, str]]:
    
    system_content = SYSTEM_PROMPT
    if knowledge:
        system_content += knowledge

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
    ]

    if history:
        for m in history:
            role = m.get("role", "user")
            text = m.get("text", "")
            if not text:
                continue
            if role not in ("user", "assistant", "system"):
                role = "user"
            messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": user_prompt})
    return messages


def generate_response_from_model(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    history: Optional[List[Dict[str, str]]] = None,
    knowledge: Optional[str] = None,
) -> Dict[str, str]:
    try:
        messages = _build_messages(prompt, history=history, knowledge=knowledge)

        print("\n")
        print("Messages:", messages)
        print("\n")
        
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        sampler = make_sampler(
            temp=0.9,
            top_p=0.8
        )
        response = generate(
            model,
            tokenizer,
            prompt=chat_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        if isinstance(response, str):
            text = response
        else:
            text = str(response)

        return {"response": text, "status": "success"}

    except Exception as e:
        return {
            "response": "",
            "status": "error",
            "error": str(e),
        }
        

def get_internal_knowledge(query: str, top_k: int) -> str:
    q_emb = embed_model.encode([query])
    if hasattr(q_emb, "tolist"):
        q_emb = q_emb.tolist()
    if not isinstance(q_emb[0], (list, tuple)):
        q_embs = [q_emb]
    else:
        q_embs = q_emb

    results = collection.query(query_embeddings=q_embs, n_results=top_k)

    docs = results.get("documents")
    if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
        docs = docs[0]

    knowledge_text = ""
    if docs:
        knowledge_text = "\n\n--- Conocimiento interno ---\n"
        for i, d in enumerate(docs[:top_k], 1):
            knowledge_text += f"[{i}] {d}\n"
            
    return knowledge_text