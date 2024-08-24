import numpy as np
from config.core import config
from pipeline import census_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    #In Research phase, it was observed that the salary has extra spaces.
    #Spaces need to be trimmed.

    y_train = np.log(y_train)

    # fit model
    census_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=census_pipe)


if __name__ == "__main__":
    run_training()