import pandas as pd

EMAILS = pd.read_csv("data/raw/email_addresses.csv", header=None, names=["email_address"])


def find_email_address(name=""):
    """
    Finds the email address of an employee by their name.

    Parameters
    ----------
    name : str
        Name of the person.

    Returns
    -------
    email_address : str
        Email addresses of the person.
    """
    global EMAILS
    if name == "":
        return "Name not provided."
    name = name.lower()
    email_address = EMAILS[EMAILS["email_address"].str.contains(name)]
    return email_address["email_address"].values

find_email_address.name = "company_directory.find_email_address"
find_email_address.func = find_email_address
find_email_address.openai_schema = {
    "type": "function",
    "function": {
        "name": "company_directory__find_email_address",
        "description": "Finds the email address of an employee by their name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the person to look up."},
            },
        },
    },
}
