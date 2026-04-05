import pandas as pd

CALENDAR_EVENTS = pd.read_csv("data/processed/calendar_events.csv", dtype=str)


def reset_state():
    global CALENDAR_EVENTS
    CALENDAR_EVENTS = pd.read_csv("data/processed/calendar_events.csv", dtype=str)


def get_event_information_by_id(event_id=None, field=None):
    if not event_id:
        return "Event ID not provided."
    if not field:
        return "Field not provided."
    event = CALENDAR_EVENTS[CALENDAR_EVENTS["event_id"] == event_id].to_dict(orient="records")
    if event:
        if field in event[0]:
            return {field: event[0][field]}
        else:
            return "Field not found."
    else:
        return "Event not found."

get_event_information_by_id.name = "calendar.get_event_information_by_id"
get_event_information_by_id.func = get_event_information_by_id
get_event_information_by_id.openai_schema = {
    "type": "function",
    "function": {
        "name": "calendar__get_event_information_by_id",
        "description": "Returns event information for a given event ID and field.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "8-digit ID of the event."},
                "field": {"type": "string", "description": "Field to return: event_id, event_name, participant_email, event_start, duration"},
            },
        },
    },
}


def search_events(query="", time_min=None, time_max=None):
    events = CALENDAR_EVENTS[
        (CALENDAR_EVENTS["event_name"].str.contains(query, case=False))
        | (CALENDAR_EVENTS["participant_email"].str.contains(query, case=False))
    ].to_dict(orient="records")
    if time_min:
        events = [e for e in events if pd.Timestamp(e["event_start"]) >= pd.Timestamp(time_min)]
    if time_max:
        events = [e for e in events if pd.Timestamp(e["event_start"]) <= pd.Timestamp(time_max)]
    return events[:5] if events else "No events found."

search_events.name = "calendar.search_events"
search_events.func = search_events
search_events.openai_schema = {
    "type": "function",
    "function": {
        "name": "calendar__search_events",
        "description": "Searches calendar events by query string and optional time bounds. Returns up to 5 events.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term matched against event_name and participant_email."},
                "time_min": {"type": "string", "description": "Lower bound for event start time. Format: YYYY-MM-DD HH:MM:SS"},
                "time_max": {"type": "string", "description": "Upper bound for event start time. Format: YYYY-MM-DD HH:MM:SS"},
            },
        },
    },
}


def create_event(event_name=None, participant_email=None, event_start=None, duration=None):
    global CALENDAR_EVENTS
    if not event_name:
        return "Event name not provided."
    if not participant_email:
        return "Participant email not provided."
    if not event_start:
        return "Event start not provided."
    if not duration:
        return "Event duration not provided."
    participant_email = participant_email.lower()
    event_id = str(int(CALENDAR_EVENTS["event_id"].max()) + 1).zfill(8)
    new_event = pd.DataFrame({
        "event_id": [event_id], "event_name": [event_name],
        "participant_email": [participant_email], "event_start": [event_start], "duration": [duration],
    })
    CALENDAR_EVENTS = pd.concat([CALENDAR_EVENTS, new_event])
    return event_id

create_event.name = "calendar.create_event"
create_event.func = create_event
create_event.openai_schema = {
    "type": "function",
    "function": {
        "name": "calendar__create_event",
        "description": "Creates a new calendar event. Returns the new event_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_name": {"type": "string", "description": "Name of the event."},
                "participant_email": {"type": "string", "description": "Email of the participant."},
                "event_start": {"type": "string", "description": "Start time. Format: YYYY-MM-DD HH:MM:SS"},
                "duration": {"type": "string", "description": "Duration in minutes."},
            },
        },
    },
}


def delete_event(event_id=None):
    global CALENDAR_EVENTS
    if not event_id:
        return "Event ID not provided."
    if event_id in CALENDAR_EVENTS["event_id"].values:
        CALENDAR_EVENTS = CALENDAR_EVENTS[CALENDAR_EVENTS["event_id"] != event_id]
        return "Event deleted successfully."
    else:
        return "Event not found."

delete_event.name = "calendar.delete_event"
delete_event.func = delete_event
delete_event.openai_schema = {
    "type": "function",
    "function": {
        "name": "calendar__delete_event",
        "description": "Deletes a calendar event by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "8-digit ID of the event."},
            },
        },
    },
}


def update_event(event_id=None, field=None, new_value=None):
    global CALENDAR_EVENTS
    if not event_id or not field or not new_value:
        return "Event ID, field, or new value not provided."
    if event_id in CALENDAR_EVENTS["event_id"].values:
        if field == "participant_email":
            new_value = new_value.lower()
        CALENDAR_EVENTS.loc[CALENDAR_EVENTS["event_id"] == event_id, field] = new_value
        return "Event updated successfully."
    else:
        return "Event not found."

update_event.name = "calendar.update_event"
update_event.func = update_event
update_event.openai_schema = {
    "type": "function",
    "function": {
        "name": "calendar__update_event",
        "description": "Updates a field on a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string", "description": "8-digit ID of the event."},
                "field": {"type": "string", "description": "Field to update: event_name, participant_email, event_start, duration"},
                "new_value": {"type": "string", "description": "New value for the field."},
            },
        },
    },
}
