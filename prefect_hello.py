from prefect import flow, task


@task
def say_hello(name):
    message = f"Hello, {name}. Prefect is working."
    print(message)
    return message


@flow(name="hello-prefect")
def hello_flow(name="Sara"):
    say_hello(name)


if __name__ == "__main__":
    hello_flow()