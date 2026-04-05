import pandas as pd
from src.data_generation.data_generation_utils import HARDCODED_CURRENT_TIME

EMAILS = pd.read_csv("data/processed/emails.csv", dtype=str)


def reset_state():
    global EMAILS
    EMAILS = pd.read_csv("data/processed/emails.csv", dtype=str)


def get_email_information_by_id(email_id=None, field=None):
    if not email_id:
        return "Email ID not provided."
    if not field:
        return "Field not provided."
    email = EMAILS[EMAILS["email_id"] == email_id].to_dict(orient="records")
    if email:
        if field in email[0]:
            return {field: email[0][field]}
        else:
            return "Field not found."
    else:
        return "Email not found."

get_email_information_by_id.name = "email.get_email_information_by_id"
get_email_information_by_id.func = get_email_information_by_id
get_email_information_by_id.openai_schema = {
    "type": "function",
    "function": {
        "name": "email__get_email_information_by_id",
        "description": "Retrieves specific details of an email by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string", "description": "Unique ID of the email."},
                "field": {"type": "string", "description": "Field to return: email_id, sender, subject, sent_date, body, inbox/outbox"},
            },
        },
    },
}


def search_emails(query="", date_min=None, date_max=None):
    query_words = query.lower().split()

    def filter_emails(row):
        combined_fields = f"{row['subject']} {row['body']} {row['sender/recipient']}".lower()
        return all(word in combined_fields for word in query_words)

    filtered_emails = EMAILS.apply(filter_emails, axis=1)
    emails = EMAILS[filtered_emails].sort_values("sent_datetime", ascending=False).to_dict(orient="records")
    if date_min:
        emails = [e for e in emails if pd.Timestamp(e["sent_datetime"]).date() >= pd.Timestamp(date_min).date()]
    if date_max:
        emails = [e for e in emails if pd.Timestamp(e["sent_datetime"]).date() <= pd.Timestamp(date_max).date()]
    return emails[:5] if emails else "No emails found."

search_emails.name = "email.search_emails"
search_emails.func = search_emails
search_emails.openai_schema = {
    "type": "function",
    "function": {
        "name": "email__search_emails",
        "description": "Searches for emails matching query across subject, body, and sender fields. Returns up to 5 results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query matched in subject, body, or sender fields."},
                "date_min": {"type": "string", "description": "Lower date limit. Format: YYYY-MM-DD"},
                "date_max": {"type": "string", "description": "Upper date limit. Format: YYYY-MM-DD"},
            },
        },
    },
}


def send_email(recipient=None, subject=None, body=None):
    global EMAILS
    if not recipient or not subject or not body:
        return "Recipient, subject, or body not provided."
    if "@" not in recipient or "." not in recipient:
        return "Invalid recipient email address."
    recipient = recipient.lower()
    email_id = str(int(EMAILS["email_id"].max()) + 1)
    sent_datetime = HARDCODED_CURRENT_TIME
    EMAILS.loc[len(EMAILS)] = [email_id, "outbox", recipient, subject, sent_datetime, body]
    return "Email sent successfully."

send_email.name = "email.send_email"
send_email.func = send_email
send_email.openai_schema = {
    "type": "function",
    "function": {
        "name": "email__send_email",
        "description": "Sends an email to the specified recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Email address of the recipient."},
                "subject": {"type": "string", "description": "Subject line of the email."},
                "body": {"type": "string", "description": "Body content of the email."},
            },
        },
    },
}


def delete_email(email_id=None):
    global EMAILS
    if not email_id:
        return "Email ID not provided."
    if email_id in EMAILS["email_id"].values:
        EMAILS = EMAILS[EMAILS["email_id"] != email_id]
        return "Email deleted successfully."
    else:
        return "Email not found."

delete_email.name = "email.delete_email"
delete_email.func = delete_email
delete_email.openai_schema = {
    "type": "function",
    "function": {
        "name": "email__delete_email",
        "description": "Deletes an email by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string", "description": "Unique ID of the email to be deleted."},
            },
        },
    },
}


def forward_email(email_id=None, recipient=None):
    global EMAILS
    if not email_id or not recipient:
        return "Email ID or recipient not provided."
    if email_id not in EMAILS["email_id"].values:
        return "Email not found."
    if "@" not in recipient or "." not in recipient:
        return "Invalid recipient email address."
    recipient = recipient.lower()
    email = EMAILS[EMAILS["email_id"] == email_id].to_dict(orient="records")[0]
    result = send_email(recipient, f"FW: {email['subject']}", email["body"])
    return "Email forwarded successfully." if result == "Email sent successfully." else result

forward_email.name = "email.forward_email"
forward_email.func = forward_email
forward_email.openai_schema = {
    "type": "function",
    "function": {
        "name": "email__forward_email",
        "description": "Forwards an email to the specified recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string", "description": "Unique ID of the email to be forwarded."},
                "recipient": {"type": "string", "description": "Email address of the recipient."},
            },
        },
    },
}


def reply_email(email_id=None, body=None):
    global EMAILS
    if not email_id or not body:
        return "Email ID or body not provided."
    if email_id not in EMAILS["email_id"].values:
        return "Email not found."
    email = EMAILS[EMAILS["email_id"] == email_id].to_dict(orient="records")[0]
    result = send_email(email["sender/recipient"], f"{email['subject']}", body)
    return "Email replied successfully." if result == "Email sent successfully." else result

reply_email.name = "email.reply_email"
reply_email.func = reply_email
reply_email.openai_schema = {
    "type": "function",
    "function": {
        "name": "email__reply_email",
        "description": "Replies to an email by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "email_id": {"type": "string", "description": "Unique ID of the email to be replied to."},
                "body": {"type": "string", "description": "Body content of the reply."},
            },
        },
    },
}
