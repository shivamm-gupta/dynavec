from unittest.mock import MagicMock

import boto3
import pytest

from dynavec.cache import DynamoDBCache
from dynavec.config import DynavecConfig
from dynavec.graph import GraphStore
from dynavec.provisioning import ensure_index, ensure_table, ensure_vector_bucket
from dynavec.stores.dynamodb import DynamoDBStore
from dynavec.stores.s3vectors import S3VectorsStore


def _config(**kwargs):
    return DynavecConfig(
        vector_bucket="bucket",
        index="index",
        table="table",
        dimension=8,
        region="us-east-1",
        **kwargs,
    )


@pytest.mark.parametrize("value", [0, -1, 1.5, "10", None, True, False])
def test_connection_pool_size_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_pool_connections must be a positive integer"):
        _config(max_pool_connections=value)


@pytest.mark.parametrize("pool_options, expected", [({}, 10), ({"max_pool_connections": 32}, 32)])
@pytest.mark.parametrize("store_type", [S3VectorsStore, DynamoDBStore, GraphStore, DynamoDBCache])
def test_store_connection_pool_configuration(store_type, pool_options, expected):
    session = boto3.Session(
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    store = store_type(_config(**pool_options), boto_session=session)
    if isinstance(store, S3VectorsStore):
        client = store._client
    else:
        client = store._table.meta.client

    try:
        assert client.meta.config.max_pool_connections == expected
        assert client.meta.region_name == "us-east-1"
    finally:
        client.close()


@pytest.mark.parametrize("pool_options, expected", [({}, 10), ({"max_pool_connections": 32}, 32)])
@pytest.mark.parametrize(
    "provision, service",
    [(ensure_vector_bucket, "s3vectors"), (ensure_index, "s3vectors"), (ensure_table, "dynamodb")],
)
def test_provisioning_connection_pool_configuration(provision, service, pool_options, expected):
    session = MagicMock()

    provision(_config(**pool_options), boto_session=session)

    session.client.assert_called_once()
    args, kwargs = session.client.call_args
    assert args == (service,)
    assert kwargs["region_name"] == "us-east-1"
    assert kwargs["config"].max_pool_connections == expected
