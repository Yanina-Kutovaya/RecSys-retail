from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email": ["kutovaiayp@yanex.ru"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}
with DAG(
    "test_dag",
    default_args=default_args,
    description="Installation DAG",
    schedule="@once",
    start_date=days_ago(2),
) as dag:
    t1 = BashOperator(
        task_id="test_git",
        bash_command='echo "Hello world!" > hello_world.txt',
    )

    t1
