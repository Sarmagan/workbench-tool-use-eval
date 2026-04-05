import pandas as pd

CRM_DATA = pd.read_csv("data/processed/customer_relationship_manager_data.csv", dtype=str)


def reset_state():
    global CRM_DATA
    CRM_DATA = pd.read_csv("data/processed/customer_relationship_manager_data.csv", dtype=str)


def search_customers(customer_name=None, customer_email=None, product_interest=None, status=None,
                     assigned_to_email=None, last_contact_date_min=None, last_contact_date_max=None,
                     follow_up_by_min=None, follow_up_by_max=None):
    customers = CRM_DATA.copy()
    if not any([customer_name, customer_email, product_interest, status, assigned_to_email,
                last_contact_date_min, last_contact_date_max, follow_up_by_min, follow_up_by_max]):
        return "No search parameters provided. Please provide at least one parameter."
    if customer_name:
        customers = customers[customers["customer_name"].str.contains(customer_name, case=False)]
    if customer_email:
        customers = customers[customers["customer_email"].str.contains(customer_email, case=False)]
    if product_interest:
        customers = customers[customers["product_interest"].str.contains(product_interest, case=False)]
    if status:
        customers = customers[customers["status"].str.contains(status, case=False)]
    if assigned_to_email:
        customers = customers[customers["assigned_to_email"].str.contains(assigned_to_email, case=False)]
    if last_contact_date_min:
        customers = customers[customers["last_contact_date"] >= last_contact_date_min]
    if last_contact_date_max:
        customers = customers[customers["last_contact_date"] <= last_contact_date_max]
    if follow_up_by_min:
        customers = customers[customers["follow_up_by"] >= follow_up_by_min]
    if follow_up_by_max:
        customers = customers[customers["follow_up_by"] <= follow_up_by_max]
    return customers.to_dict(orient="records")[:5]

search_customers.name = "customer_relationship_manager.search_customers"
search_customers.func = search_customers
search_customers.openai_schema = {
    "type": "function",
    "function": {
        "name": "customer_relationship_manager__search_customers",
        "description": "Searches for customers based on given parameters. Returns up to 5 records.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string", "description": "Name of the customer."},
                "customer_email": {"type": "string", "description": "Email address of the customer."},
                "product_interest": {"type": "string", "description": "Product interest of the customer."},
                "status": {"type": "string", "description": "Customer status: Qualified, Won, Lost, Lead, Proposal"},
                "assigned_to_email": {"type": "string", "description": "Email of the assigned team member."},
                "last_contact_date_min": {"type": "string", "description": "Min last contact date. Format: YYYY-MM-DD"},
                "last_contact_date_max": {"type": "string", "description": "Max last contact date. Format: YYYY-MM-DD"},
                "follow_up_by_min": {"type": "string", "description": "Min follow-up date. Format: YYYY-MM-DD"},
                "follow_up_by_max": {"type": "string", "description": "Max follow-up date. Format: YYYY-MM-DD"},
            },
        },
    },
}


def update_customer(customer_id=None, field=None, new_value=None):
    global CRM_DATA
    if not customer_id or not field or not new_value:
        return "Customer ID, field, or new value not provided."
    if field == "status" and new_value not in ["Qualified", "Won", "Lost", "Lead", "Proposal"]:
        return "Status not valid. Please choose from: 'Qualified', 'Won', 'Lost', 'Lead', 'Proposal'"
    if field == "product_interest" and new_value not in ["Software", "Hardware", "Services", "Consulting", "Training"]:
        return "Product interest not valid. Please choose from: 'Software', 'Hardware', 'Services', 'Consulting', 'Training'"
    if field in ["customer_email", "assigned_to_email"]:
        new_value = new_value.lower()
    if customer_id in CRM_DATA["customer_id"].values:
        if field in CRM_DATA.columns:
            CRM_DATA.loc[CRM_DATA["customer_id"] == customer_id, field] = new_value
            return "Customer updated successfully."
        else:
            return "Field not valid."
    else:
        return "Customer not found."

update_customer.name = "customer_relationship_manager.update_customer"
update_customer.func = update_customer
update_customer.openai_schema = {
    "type": "function",
    "function": {
        "name": "customer_relationship_manager__update_customer",
        "description": "Updates a customer record field by customer ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "ID of the customer."},
                "field": {"type": "string", "description": "Field to update: customer_name, assigned_to_email, customer_email, customer_phone, last_contact_date, product_interest, status, notes, follow_up_by"},
                "new_value": {"type": "string", "description": "New value for the field."},
            },
        },
    },
}


def add_customer(customer_name=None, assigned_to_email=None, status=None, customer_email=None,
                 customer_phone=None, last_contact_date=None, product_interest=None, notes="", follow_up_by=None):
    global CRM_DATA
    if not all([customer_name, assigned_to_email, status]):
        return "Please provide all required fields: customer_name, assigned_to_email, status."
    assigned_to_email = assigned_to_email.lower()
    if customer_email:
        customer_email = customer_email.lower()
    new_id = str(int(CRM_DATA["customer_id"].max()) + 1).zfill(8)
    new_customer = pd.DataFrame({
        "customer_id": [new_id], "customer_name": [customer_name], "customer_email": [customer_email],
        "customer_phone": [customer_phone], "last_contact_date": [last_contact_date],
        "product_interest": [product_interest], "status": [status], "assigned_to_email": [assigned_to_email],
        "notes": [notes], "follow_up_by": [follow_up_by],
    })
    CRM_DATA = pd.concat([CRM_DATA, new_customer], ignore_index=True)
    return new_id

add_customer.name = "customer_relationship_manager.add_customer"
add_customer.func = add_customer
add_customer.openai_schema = {
    "type": "function",
    "function": {
        "name": "customer_relationship_manager__add_customer",
        "description": "Adds a new customer record. Returns the new customer_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string", "description": "Name of the customer."},
                "assigned_to_email": {"type": "string", "description": "Email of the assigned team member."},
                "status": {"type": "string", "description": "Status: Qualified, Won, Lost, Lead, Proposal"},
                "customer_email": {"type": "string", "description": "Email of the customer."},
                "customer_phone": {"type": "string", "description": "Phone number."},
                "last_contact_date": {"type": "string", "description": "Last contact date. Format: YYYY-MM-DD"},
                "product_interest": {"type": "string", "description": "Product interest: Software, Hardware, Services, Consulting, Training"},
                "notes": {"type": "string", "description": "Notes about the customer."},
                "follow_up_by": {"type": "string", "description": "Follow-up date. Format: YYYY-MM-DD"},
            },
        },
    },
}


def delete_customer(customer_id=None):
    global CRM_DATA
    if not customer_id:
        return "Customer ID not provided."
    if customer_id not in CRM_DATA["customer_id"].values:
        return "Customer not found."
    CRM_DATA = CRM_DATA[CRM_DATA["customer_id"] != customer_id]
    return "Customer deleted successfully."

delete_customer.name = "customer_relationship_manager.delete_customer"
delete_customer.func = delete_customer
delete_customer.openai_schema = {
    "type": "function",
    "function": {
        "name": "customer_relationship_manager__delete_customer",
        "description": "Deletes a customer record by ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "ID of the customer."},
            },
        },
    },
}
