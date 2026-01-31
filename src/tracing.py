import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace

# Global tracer
tracer = None

def setup_tracing():
    global tracer
    
    # Ρύθμιση Phoenix (Τοπικά)
    os.environ["PHOENIX_PROJECT_NAME"] = "support-bot-v1"
    
    # Καταγραφή (Tracing Provider)
    tracer_provider = register(
        project_name="support-bot-v1",
        endpoint="http://localhost:6006/v1/traces"
    )
    
    # Αυτόματη σύνδεση με OpenAI
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    
    # Δημιουργία Tracer
    tracer = trace.get_tracer("support-agent")
    
    return tracer

def get_tracer():
    if tracer is None:
        return trace.get_tracer("support-agent")
    return tracer