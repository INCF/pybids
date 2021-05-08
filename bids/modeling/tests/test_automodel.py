from os.path import join
from bids.modeling.auto_model import auto_model
from bids.modeling import BIDSStatsModelsGraph
from bids.layout import BIDSLayout
from bids.tests import get_test_data_path
import pytest


@pytest.fixture
def model():
    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)

    models = auto_model(layout, scan_length=480, one_vs_rest=True)

    return models[0]


def test_auto_model_graph(model):

    layout_path = join(get_test_data_path(), 'ds005')
    layout = BIDSLayout(layout_path)

    # Test to make sure an analaysis can be setup from the generated model
    graph = BIDSStatsModelsGraph(layout, model)
    graph.load_collections(scan_length=480)

    assert model['Name'] == 'ds005_mixedgamblestask'

    # run level
    block = model['Nodes'][0]
    assert block['Name'] == 'Run'
    assert block['Level'] == 'Run'
    assert block['Transformations'][0]['Name'] == 'Factor'
    assert block['Contrasts'][0]['Name'] == 'run_parametric gain'
    assert block['Contrasts'][0]['Weights'] == [1]
    assert block['Contrasts'][0]['Type'] == 't'

    # subject level
    block = model['Nodes'][1]
    assert block['Name'] == 'Subject'
    assert block['Level'] == 'Subject'
    assert block['Model']['X'][0] == 'run_parametric gain'
    assert block['Contrasts'][0]['Name'] == 'subject_run_parametric gain'
    assert block['Contrasts'][0]['Type'] == 'FEMA'

    # dataset level
    block = model['Nodes'][2]
    assert block['Name'] == 'Dataset'
    assert block['Level'] == 'Dataset'
    assert block['Model']['X'][0] == 'subject_run_parametric gain'
    assert block['Contrasts'][0]['Name'] == 'dataset_subject_run_parametric gain'
    assert block['Contrasts'][0]['Type'] == 't'
