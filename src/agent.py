import json
from openai import OpenAI
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes
from src.tracing import get_tracer
from src.data import ORDER_DATABASE, FAQ_DATABASE, cosine_similarity, initialize_faq_embeddings

client = OpenAI()
tracer = get_tracer()

# --- Tools Definition ---
tools = [{
    "type": "function",
    "function": {
        "name": "lookupOrderStatus",
        "description": "Look up the current status of a customer order by order ID",
        "parameters": {
            "type": "object",
            "properties": {
                "orderId": {"type": "string", "description": "The order ID (e.g., ORD-12345)"}
            },
            "required": ["orderId"],
        },
    },
}]

def handle_support_query(user_query: str):
    with tracer.start_as_current_span("support-agent", attributes={SpanAttributes.INPUT_VALUE: user_query}) as agent_span:
        
        print(f"\n🤖 Processing: '{user_query}'")

        # 1. Classification (Router)
        classification_prompt = """Classify the query into: 
        1. "order_status" (tracking, shipping, ETA)
        2. "faq" (refunds, password, billing)
        Respond JSON: {"category": "...", "reasoning": "..."}"""
        
        cls_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"}
        )
        cls_data = json.loads(cls_response.choices[0].message.content)
        category = cls_data.get("category", "faq")
        print(f"📋 Category: {category} ({cls_data.get('reasoning')})")

        # 2. Logic Branching
        if category == "order_status":
            return handle_order_flow(user_query)
        else:
            return handle_rag_flow(user_query)

def handle_order_flow(user_query):
    """Χειρίζεται τις παραγγελίες καλώντας Tools"""
    print("🔧 Branch: Tool Execution")
    
    # Ask LLM to use tool
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a support agent. Use lookupOrderStatus if an ID is present."},
            {"role": "user", "content": user_query}
        ],
        tools=tools,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        order_id = args.get("orderId")
        
        print(f"🔧 Tool Called: lookupOrderStatus({order_id})")
        
        # Simulate DB Lookup with Span
        with tracer.start_as_current_span("lookupOrderStatus") as tool_span:
            order_data = ORDER_DATABASE.get(order_id, {"error": "Order not found"})
            tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(order_data))
        
        # Final Response
        final_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the order status nicely."},
                {"role": "user", "content": f"Query: {user_query}. Data: {json.dumps(order_data)}"}
            ]
        )
        return final_res.choices[0].message.content
    else:
        return "Please provide a valid Order ID (e.g., ORD-12345)."

def handle_rag_flow(user_query):
    """Χειρίζεται τα FAQs με RAG"""
    print("📚 Branch: RAG Knowledge Base")
    
    # 1. Embed Query
    q_embed = client.embeddings.create(model="text-embedding-ada-002", input=user_query).data[0].embedding
    
    # 2. Find Similar
    scores = []
    for faq in FAQ_DATABASE:
        if faq["embedding"]:
            s = cosine_similarity(q_embed, faq["embedding"])
            scores.append((faq, s))
            
    top_faqs = sorted(scores, key=lambda x: x[1], reverse=True)[:2]
    
    # 3. Create Context
    context = "\n".join([f"Q: {f['question']}\nA: {f['answer']}" for f, s in top_faqs])
    
    # 4. Generate Answer
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Answer based on context:\n{context}"},
            {"role": "user", "content": user_query}
        ]
    )
    return res.choices[0].message.content