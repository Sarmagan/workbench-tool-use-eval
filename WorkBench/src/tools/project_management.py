import pandas as pd

PROJECT_TASKS = pd.read_csv("data/processed/project_tasks.csv", dtype=str)


def reset_state():
    global PROJECT_TASKS
    PROJECT_TASKS = pd.read_csv("data/processed/project_tasks.csv", dtype=str)


def get_task_information_by_id(task_id=None, field=None):
    if not task_id:
        return "Task ID not provided."
    if not field:
        return "Field not provided."
    task = PROJECT_TASKS[PROJECT_TASKS["task_id"] == task_id].to_dict(orient="records")
    if task:
        if field in task[0]:
            return {field: task[0][field]}
        else:
            return "Field not found."
    else:
        return "Task not found."

get_task_information_by_id.name = "project_management.get_task_information_by_id"
get_task_information_by_id.func = get_task_information_by_id
get_task_information_by_id.openai_schema = {
    "type": "function",
    "function": {
        "name": "project_management__get_task_information_by_id",
        "description": "Returns task information for a given task ID and field.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "8-digit ID of the task."},
                "field": {"type": "string", "description": "Field to return: task_id, task_name, assigned_to_email, list_name, due_date, board"},
            },
        },
    },
}


def search_tasks(task_name=None, assigned_to_email=None, list_name=None, due_date=None, board=None):
    if not any([task_name, assigned_to_email, list_name, due_date, board]):
        return "No search parameters provided."
    tasks = PROJECT_TASKS.copy()
    if task_name:
        tasks = tasks[tasks["task_name"].str.contains(task_name, case=False)]
    if assigned_to_email:
        tasks = tasks[tasks["assigned_to_email"].str.contains(assigned_to_email, case=False)]
    if list_name:
        tasks = tasks[tasks["list_name"].str.contains(list_name, case=False)]
    if due_date:
        tasks = tasks[tasks["due_date"].str.contains(due_date, case=False)]
    if board:
        tasks = tasks[tasks["board"].str.contains(board, case=False)]
    return tasks.to_dict(orient="records")

search_tasks.name = "project_management.search_tasks"
search_tasks.func = search_tasks
search_tasks.openai_schema = {
    "type": "function",
    "function": {
        "name": "project_management__search_tasks",
        "description": "Searches for tasks based on given parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Name of the task."},
                "assigned_to_email": {"type": "string", "description": "Email of the assignee."},
                "list_name": {"type": "string", "description": "List name: Backlog, In Progress, In Review, Completed"},
                "due_date": {"type": "string", "description": "Due date. Format: YYYY-MM-DD"},
                "board": {"type": "string", "description": "Board name: Back end, Front end, Design"},
            },
        },
    },
}


def create_task(task_name=None, assigned_to_email=None, list_name=None, due_date=None, board=None):
    global PROJECT_TASKS
    if not all([task_name, assigned_to_email, list_name, due_date, board]):
        return "Missing task details."
    assigned_to_email = assigned_to_email.lower()
    if assigned_to_email not in PROJECT_TASKS["assigned_to_email"].str.lower().values:
        return "Assignee email not valid. Please choose from the list of team members."
    if list_name not in ["Backlog", "In Progress", "In Review", "Completed"]:
        return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
    if board not in ["Back end", "Front end", "Design"]:
        return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."
    task_id = str(int(PROJECT_TASKS["task_id"].max()) + 1).zfill(8)
    new_task = pd.DataFrame({
        "task_id": [task_id], "task_name": [task_name], "assigned_to_email": [assigned_to_email],
        "list_name": [list_name], "due_date": [due_date], "board": [board],
    })
    PROJECT_TASKS = pd.concat([PROJECT_TASKS, new_task], ignore_index=True)
    return task_id

create_task.name = "project_management.create_task"
create_task.func = create_task
create_task.openai_schema = {
    "type": "function",
    "function": {
        "name": "project_management__create_task",
        "description": "Creates a new project task. Returns the new task_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Name of the task."},
                "assigned_to_email": {"type": "string", "description": "Email of the assignee."},
                "list_name": {"type": "string", "description": "List: Backlog, In Progress, In Review, Completed"},
                "due_date": {"type": "string", "description": "Due date. Format: YYYY-MM-DD"},
                "board": {"type": "string", "description": "Board: Back end, Front end, Design"},
            },
        },
    },
}


def delete_task(task_id=None):
    global PROJECT_TASKS
    if not task_id:
        return "Task ID not provided."
    if task_id in PROJECT_TASKS["task_id"].values:
        PROJECT_TASKS = PROJECT_TASKS[PROJECT_TASKS["task_id"] != task_id]
        return "Task deleted successfully."
    else:
        return "Task not found."

delete_task.name = "project_management.delete_task"
delete_task.func = delete_task
delete_task.openai_schema = {
    "type": "function",
    "function": {
        "name": "project_management__delete_task",
        "description": "Deletes a task by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "8-digit ID of the task."},
            },
        },
    },
}


def update_task(task_id=None, field=None, new_value=None):
    global PROJECT_TASKS
    if not task_id or not field or not new_value:
        return "Task ID, field, or new value not provided."
    if field == "assigned_to_email":
        new_value = new_value.lower()
    if field == "board" and new_value not in ["Back end", "Front end", "Design"]:
        return "Board not valid. Please choose from: 'Back end', 'Front end', 'Design'."
    if field == "list_name" and new_value not in ["Backlog", "In Progress", "In Review", "Completed"]:
        return "List not valid. Please choose from: 'Backlog', 'In Progress', 'In Review', 'Completed'."
    if field == "assigned_to_email" and new_value not in PROJECT_TASKS["assigned_to_email"].str.lower().values:
        return "Assignee email not valid. Please choose from the list of team members."
    if task_id in PROJECT_TASKS["task_id"].values:
        if field in PROJECT_TASKS.columns:
            PROJECT_TASKS.loc[PROJECT_TASKS["task_id"] == task_id, field] = new_value
            return "Task updated successfully."
        else:
            return "Field not valid."
    else:
        return "Task not found."

update_task.name = "project_management.update_task"
update_task.func = update_task
update_task.openai_schema = {
    "type": "function",
    "function": {
        "name": "project_management__update_task",
        "description": "Updates a field on a task.",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "8-digit ID of the task."},
                "field": {"type": "string", "description": "Field to update: task_name, assigned_to_email, list_name, due_date, board"},
                "new_value": {"type": "string", "description": "New value for the field."},
            },
        },
    },
}
