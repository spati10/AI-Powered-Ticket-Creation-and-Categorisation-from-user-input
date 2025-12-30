from .predict import predict_ticket
from datetime import datetime
import uuid

def generate_ticket(text):
    category, priority = predict_ticket(text)

    ticket = {
        "ticket_id": str(uuid.uuid4())[:8],
        "description": text,
        "category": category,
        "priority": priority,
        "status": "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return ticket

if __name__ == "__main__":
    ticket = generate_ticket("payment is not showing")
    print(ticket)
