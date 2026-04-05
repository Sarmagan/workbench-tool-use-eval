import pandas as pd

ANALYTICS_DATA = pd.read_csv("data/processed/analytics_data.csv", dtype=str)
ANALYTICS_DATA["user_engaged"] = ANALYTICS_DATA["user_engaged"] == "True"
PLOTS_DATA = pd.DataFrame(columns=["file_path"])
METRICS = ["total_visits", "session_duration_seconds", "user_engaged"]
METRIC_NAMES = ["total visits", "average session duration", "engaged users"]


def reset_state():
    global ANALYTICS_DATA, PLOTS_DATA
    ANALYTICS_DATA = pd.read_csv("data/processed/analytics_data.csv", dtype=str)
    ANALYTICS_DATA["user_engaged"] = ANALYTICS_DATA["user_engaged"] == "True"
    PLOTS_DATA = pd.DataFrame(columns=["file_path"])


def get_visitor_information_by_id(visitor_id=None):
    if not visitor_id:
        return "Visitor ID not provided."
    visitor_data = ANALYTICS_DATA[ANALYTICS_DATA["visitor_id"] == visitor_id].to_dict(orient="records")
    return visitor_data if visitor_data else "Visitor not found."

get_visitor_information_by_id.name = "analytics.get_visitor_information_by_id"
get_visitor_information_by_id.func = get_visitor_information_by_id
get_visitor_information_by_id.openai_schema = {
    "type": "function",
    "function": {
        "name": "analytics__get_visitor_information_by_id",
        "description": "Returns the analytics data for a given visitor ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "visitor_id": {"type": "string", "description": "ID of the visitor."},
            },
        },
    },
}


def create_plot(time_min=None, time_max=None, value_to_plot=None, plot_type=None):
    global PLOTS_DATA
    if not time_min:
        return "Start date not provided."
    if not time_max:
        return "End date not provided."
    valid_values = ["total_visits", "session_duration_seconds", "user_engaged", "visits_direct",
                    "visits_referral", "visits_search_engine", "visits_social_media"]
    if value_to_plot not in valid_values:
        return "Value to plot must be one of 'total_visits', 'session_duration_seconds', 'user_engaged', 'direct', 'referral', 'search engine', 'social media'"
    if plot_type not in ["bar", "line", "scatter", "histogram"]:
        return "Plot type must be one of 'bar', 'line', 'scatter', or 'histogram'"
    file_path = f"plots/{time_min}_{time_max}_{value_to_plot}_{plot_type}.png"
    PLOTS_DATA.loc[len(PLOTS_DATA)] = [file_path]
    return file_path

create_plot.name = "analytics.create_plot"
create_plot.func = create_plot
create_plot.openai_schema = {
    "type": "function",
    "function": {
        "name": "analytics__create_plot",
        "description": "Creates an analytics plot for a given time range and metric.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "Start date. Format: YYYY-MM-DD"},
                "time_max": {"type": "string", "description": "End date. Format: YYYY-MM-DD"},
                "value_to_plot": {"type": "string", "description": "Metric to plot: total_visits, session_duration_seconds, user_engaged, visits_direct, visits_referral, visits_search_engine, visits_social_media"},
                "plot_type": {"type": "string", "description": "Chart type: bar, line, scatter, histogram"},
            },
        },
    },
}


def total_visits_count(time_min=None, time_max=None):
    data = ANALYTICS_DATA[ANALYTICS_DATA["date_of_visit"] >= time_min] if time_min else ANALYTICS_DATA
    if time_max:
        data = data[data["date_of_visit"] <= time_max]
    return data.groupby("date_of_visit").size().to_dict()

total_visits_count.name = "analytics.total_visits_count"
total_visits_count.func = total_visits_count
total_visits_count.openai_schema = {
    "type": "function",
    "function": {
        "name": "analytics__total_visits_count",
        "description": "Returns total number of visits per day within a time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "Start date. Format: YYYY-MM-DD"},
                "time_max": {"type": "string", "description": "End date. Format: YYYY-MM-DD"},
            },
        },
    },
}


def engaged_users_count(time_min=None, time_max=None):
    data = ANALYTICS_DATA[ANALYTICS_DATA["date_of_visit"] >= time_min] if time_min else ANALYTICS_DATA[:]
    if time_max:
        data = data[data["date_of_visit"] <= time_max]
    data["user_engaged"] = data["user_engaged"].astype(bool).astype(int)
    return data.groupby("date_of_visit").sum()["user_engaged"].to_dict()

engaged_users_count.name = "analytics.engaged_users_count"
engaged_users_count.func = engaged_users_count
engaged_users_count.openai_schema = {
    "type": "function",
    "function": {
        "name": "analytics__engaged_users_count",
        "description": "Returns number of engaged users per day within a time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "Start date. Format: YYYY-MM-DD"},
                "time_max": {"type": "string", "description": "End date. Format: YYYY-MM-DD"},
            },
        },
    },
}


def traffic_source_count(time_min=None, time_max=None, traffic_source=None):
    data = ANALYTICS_DATA[ANALYTICS_DATA["date_of_visit"] >= time_min] if time_min else ANALYTICS_DATA[:]
    if time_max:
        data = data[data["date_of_visit"] <= time_max]
    if traffic_source:
        data["visits_from_source"] = (data["traffic_source"] == traffic_source).astype(int)
        return data.groupby("date_of_visit").sum()["visits_from_source"].to_dict()
    else:
        return data.groupby("date_of_visit").size().to_dict()

traffic_source_count.name = "analytics.traffic_source_count"
traffic_source_count.func = traffic_source_count
traffic_source_count.openai_schema = {
    "type": "function",
    "function": {
        "name": "analytics__traffic_source_count",
        "description": "Returns number of visits from a specific traffic source per day.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "Start date. Format: YYYY-MM-DD"},
                "time_max": {"type": "string", "description": "End date. Format: YYYY-MM-DD"},
                "traffic_source": {"type": "string", "description": "Traffic source: direct, referral, search engine, social media"},
            },
        },
    },
}


def get_average_session_duration(time_min=None, time_max=None):
    data = ANALYTICS_DATA[ANALYTICS_DATA["date_of_visit"] >= time_min] if time_min else ANALYTICS_DATA
    if time_max:
        data = data[data["date_of_visit"] <= time_max]
    data["session_duration_seconds"] = data["session_duration_seconds"].astype(float)
    return data[["date_of_visit", "session_duration_seconds"]].groupby("date_of_visit").mean()["session_duration_seconds"].to_dict()

get_average_session_duration.name = "analytics.get_average_session_duration"
get_average_session_duration.func = get_average_session_duration
get_average_session_duration.openai_schema = {
    "type": "function",
    "function": {
        "name": "analytics__get_average_session_duration",
        "description": "Returns average session duration in seconds per day within a time range.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "Start date. Format: YYYY-MM-DD"},
                "time_max": {"type": "string", "description": "End date. Format: YYYY-MM-DD"},
            },
        },
    },
}
