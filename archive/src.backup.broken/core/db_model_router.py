class RoutingDecision:
    def __init__(self):
        self.model = "llama3.2:latest"
        self.complexity_score = 50
        self.intent = "general"
        self.domain = "general"
        self.temperature = 0.7
        self.max_tokens = 2000
        self.reasoning = "Direct routing to available model"
        self.tier = 1
        self.specialization = None
        self.requires_context = False
        self.confidence = 0.8
        # Add any other attributes that might be expected
        self.escalation_level = 0
        self.model_category = "general"

class IntelligentDatabaseRouter:
    def route_query(self, query, conversation_history=None):
        return RoutingDecision()
    
    def log_performance(self, *args, **kwargs):
        pass  # Accept any arguments

intelligent_router = IntelligentDatabaseRouter()
