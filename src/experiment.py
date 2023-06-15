import os
import sys

import hydra
import hydra.utils as hut
from omegaconf import DictConfig
from sklearn.model_selection import TimeSeriesSplit

import utils.date as dut
from data.contracts import Columns

from qs.experiment import Experiment


@hydra.main(config_path="config", config_name="config")
def main(conf: DictConfig):
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")

    print(f"Experiment train periods: {str(conf.experiment_configuration.train.periods)}")

    # print(f"Data store is {conf.data.data_store.type}")
    # print(f"Experiment: {str(conf['experiment'])}")
    pipeline = hut.instantiate(conf.ml_feature_transformer)
    print(f'Instantiated transformers: {len(pipeline.transformers)}')

    model:ml.BaseMlModel = None # type: ignore

    data_source = hut.instantiate(conf.data_source)
    print(f'Instantiated data source: {data_source.data_store_path}')

    # Train (in periods)
    for period in conf.experiment_configuration.train.periods:

        from_date, to_date = period.from_date, period.to_date
        print(f'Beginning experiment between: {from_date} and {to_date} ({type(data_source)})..')

        data = data_source.get_data(from_date, to_date)
        print(f'Beginning experiment between: {from_date} and {to_date} - training data loaded..')

        data, label = pipeline.transform(data)
        print(f'Training data transformed. Label: {label}')

        print(data.tail(10))
        print('************************')

        # NOTE: when in loop, instantiate only if none
        if not model: model = hut.instantiate(conf.ml_model)
        model.label = label # NOTE: override this according to the transformed value
        print(f'ML model instantiated: {model.model_name} (label: {model.label})')

        model_saved_to = ''
        splits = TimeSeriesSplit(n_splits=conf.experiment_configuration.train.timeseries_splits)
        for train_index, test_index in splits.split(data):
            train = data.iloc[train_index.tolist()]
            test = data.iloc[test_index.tolist()]

            model.fit(train, test)

            models_folder = conf.experiment_configuration.train.models_folder 
            model_saved_to = model.save(models_folder)
        print(f'ML model trained and saved to: {model_saved_to}')

    ## Test
    from_date, to_date = conf.experiment_configuration.experiment.from_date, conf.experiment_configuration.experiment.to_date
    prefetch_date = dut.to_date(dut.adjust_days(from_date, -dut.one_year))
    data = data_source.get_data(prefetch_date, to_date)

    data, label = pipeline.transform(data)
    print(f'Beginning experiment between: {from_date} and {to_date} - test data loaded..')
    
    # NOTE: the ML-model instantation couldn't easily fit with Hydra - see hack below
    #       Furthermore: assuming ML-model as the 1st transformer (otherwise: fails with `invalid-features` error)
    result_transformer = hut.instantiate(conf.result_transformer)
    result_transformer.transformers[0].ml_model = model # HACK replace this after it's configured
    
    data = result_transformer.transform(data)
    print(f'Result data transformed')

    ## NOTE: somehow instantiating experiment (requires data) doesn't work:
    ##   experiment = hut.instantiate(conf.experiment_configuration.experiment, data)
    ## Error locating target 'qs.experiment.Experiment', set env var HYDRA_FULL_ERROR=1 to see chained exception.
    ### could be: ModuleNotFoundError: No module named 'qstrader'
    ## HINT: instantiate experiment by hand instead
    experiment_kwargs = conf.experiment_configuration.experiment_kwargs    
    universe=hut.instantiate(
        conf.experiment_configuration.universe,
        score_data=data,
        **experiment_kwargs
        )
    print(f'Instantiated universe (v2): {type(universe)}')
    
    data_source=hut.instantiate(
        conf.experiment_configuration.data_source,
        prices=data[Columns.PRICE], 
        returns=data[Columns.RETURN],
        # **experiment_kwargs
        )
    print(f'Instantiated data_source (v2): {type(data_source)}')
    
    alpha_model=hut.instantiate(
        conf.experiment_configuration.alpha_model,
        score_data=data,
        universe=universe,
        data_source=data_source,
        **experiment_kwargs
        )
    print(f'Instantiated alpha_model (v2): {type(alpha_model)}')

    risk_model=hut.instantiate(
        conf.experiment_configuration.risk_model,
        **experiment_kwargs
    )
    print(f'Instantiated risk model (v2): {type(risk_model)}')
    
    experiment = hut.instantiate(
        conf.experiment_configuration.experiment,
        data=data,
        universe=universe,
        data_source=data_source,
        alpha_model=alpha_model,
        risk_model=risk_model,
        **experiment_kwargs
    )
    print(f'Instantiated experiment (v2): {type(experiment)}')

    experiment.run()    
    print(f'Experiment finished!')

    experiment.plot_results()
    

if __name__ == "__main__":
    main()