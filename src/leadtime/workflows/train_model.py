from typing import Tuple
from os.path import join, dirname
from ray.data import Dataset, read_parquet
from ray.data.preprocessors import Categorizer, Chain
from ray.train.sklearn import SklearnTrainer
from prefect import flow, task
from prefect_ray import RayTaskRunner
from sklearn.ensemble import RandomForestRegressor


@task
def prepare_dataset(input_folder: str) -> Tuple[Dataset, Dataset]:
    ds = read_parquet(input_folder)
    ds_train, ds_valid = ds.train_test_split(test_size=0.2, shuffle=True)

    return ds_train, ds_valid


@task
def train(ds_train: Dataset, ds_valid: Dataset) -> None:
    categorical_features = [
        "TYPE_WACHTTIJD",
        "SPECIALISME",
        "ROAZ_REGIO",
        "TYPE_ZORGINSTELLING",
    ]

    preprocessor = Chain(
        Categorizer(columns=categorical_features)
    )

    trainer = SklearnTrainer(
        estimator=RandomForestRegressor(),
        label_column='WACHTTIJD',
        datasets={"train": ds_train, "valid": ds_valid},
        preprocessor=preprocessor
    )

    result = trainer.fit()


@flow(name='train-model', task_runner=RayTaskRunner)
def train_model(input_folder: str) -> None:
    ds_train, ds_valid = prepare_dataset(input_folder)
    train(ds_train, ds_valid)


if __name__ == "__main__":
    root_folder = dirname(dirname(dirname(dirname(__file__))))
    input_file = join(root_folder, 'data/preprocessed')

    train_model(input_file)
