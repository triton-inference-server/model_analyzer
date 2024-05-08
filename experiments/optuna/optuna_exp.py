import optuna

current_best_value = 0
stagnation_count = 0


def objective(trial):
    best_value = 0
    instance_count = trial.suggest_int("instance_count", 1, 8)
    # batch_size = trial.suggest_categorical(
    #     "batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # )
    # concurrency = trial.suggest_categorical(
    #     "concurrency", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # )

    batch_size = trial.suggest_int("batch_size", 1, 10)
    # concurrency = trial.suggest_int("concurrency", 1, 10)
    concurrency = (batch_size**2 * instance_count) * 2
    size = trial.suggest_categorical("size", ["FP8", "FP16", "FP32"])

    throughput = calculate_throughput(instance_count, batch_size, concurrency, size)

    check_for_early_terminate_condition()

    return throughput


def calculate_throughput(instance_count, batch_size, concurrency, size):
    if size == "FP8":
        throughput = instance_count * (2**batch_size) - concurrency
    elif size == "FP16":
        throughput = instance_count * ((2**batch_size) / 2) + concurrency
    elif size == "FP32":
        throughput = (instance_count / 2) * ((2**batch_size) / 2) * (concurrency / 32)

    return throughput


def check_for_early_terminate_condition():
    global stagnation_count
    global current_best_value
    if study.best_trials and study.best_value > current_best_value:
        current_best_value = study.best_value
        stagnation_count = 0
    else:
        stagnation_count = stagnation_count + 1

    if stagnation_count == 50:
        print("Stopping study...")
        study.stop()


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # The optimization finishes after evaluating 1000 times or 300 seconds.
    study.optimize(objective, n_trials=500, timeout=300)
    print(f"Best params is {study.best_params} with value {study.best_value}")
