from os.path import dirname, join
import dask.dataframe as dd
from prefect import flow, task
from prefect_ray import RayTaskRunner
from ray.util.dask import enable_dask_on_ray


@task(name='load-dataset')
def load_dataset(input_file: str) -> dd.DataFrame:
    df = dd.read_csv(input_file, encoding='iso-8859-1',
                     sep=';', dtype={'WACHTTIJD': 'float64'})

    return df


@task(name='select-features')
def select_features(df: dd.DataFrame) -> dd.DataFrame:
    feature_names = [
        'TYPE_WACHTTIJD',
        'SPECIALISME',
        'ROAZ_REGIO',
        'TYPE_ZORGINSTELLING',
        'WACHTTIJD'
    ]

    return df[feature_names]


@task(name='drop-invalid-records')
def drop_invalid_records(df: dd.DataFrame) -> dd.DataFrame:
    return df.dropna(subset=['WACHTTIJD'])


@task(name='fill-missing-values')
def fill_missing_values(df: dd.DataFrame) -> dd.DataFrame:
    df['TYPE_ZORGINSTELLING'] = df['TYPE_ZORGINSTELLING'].fillna('Kliniek')
    return df


@flow(task_runner=RayTaskRunner)
def prepare_data(input_file: str, output_file: str) -> None:
    enable_dask_on_ray()

    dataset = load_dataset(input_file)
    dataset = select_features(dataset)
    dataset = drop_invalid_records(dataset)
    dataset = fill_missing_values(dataset)

    dataset.to_parquet(output_file)


if __name__ == "__main__":
    root_folder = dirname(dirname(dirname(dirname(__file__))))
    input_file = join(root_folder, 'data/raw/lead-times.csv')
    output_file = join(root_folder, 'data/preprocessed/leadtime.parquet')

    prepare_data(input_file, output_file)
