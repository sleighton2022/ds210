from fastapi.testclient import TestClient
from src.main import app
import random
from datetime import datetime
import pytest


#client = TestClient(app)

"""
@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as client:  # Use "with" statement
        yield client  # Yield the client to the test function
"""
@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as lifespanned_client:  # Use "with" statement
        yield lifespanned_client  # Yield the client to the test function

@pytest.mark.asyncio
async def test_main_lifespan_and_model_loading(client):  # Use the client fixture
    # The lifespan events (main_lifespan and lifespan_mechanism) 
    # should now be triggered before this test function runs

    response = client.get("/lab/health")
    assert response.status_code == 200

def test_lab1_root(client):
    response = client.get("/")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}

def test_lab2_health(client):
    response = client.get("/lab/health")
    assert response.status_code == 200
    assert datetime.fromisoformat(response.json()["time"])


@pytest.mark.parametrize(
    "query_parameter, value",
    [("bob", "name"), ("nam", "name")],
)
def test_lab1_hello_endpoint_bad_parameter(client, query_parameter, value):
    response = client.get(f"/lab/hello?{query_parameter}={value}")
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "input": None,
                "loc": ["query", "name"],
                "msg": "Field required",
                "type": "missing",
            }
        ]
    }


@pytest.mark.parametrize(
    "test_input, expected",
    [("james", "james"), ("bOB", "bOB"), ("BoB", "BoB"), (100, 100)],
)
def test_lab1_hello_endpoint(client, test_input, expected):
    response = client.get(f"/lab/hello?name={test_input}")
    assert response.status_code == 200
    assert response.json()["message"].lower() == f"Hello {expected}".lower()


def test_lab1_docs_endpoint(client):
    response = client.get("/lab/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_lab1_openapi_version_correct(client):
    response = client.get("/lab/openapi.json")
    assert response.status_code == 200
    assert response.json()["openapi"][0:2] == "3."
    assert response.headers["content-type"] == "application/json"


def test_lab1_hello_multiple_parameter_with_good_and_bad(client):
    response = client.get("/lab/hello?name=james&bob=name")
    assert response.status_code == 200
    assert response.json()["message"].lower() == "Hello james".lower()


# Are we able to make a basic prediction?
# Do I return the type I expect?
# we test the predition is only a particular type because model weights change as we retrain
# when model weights change we also change our results
# recommend to test the fundamental expectation and not the particular value
def test_predict_basic(client):
    data = {
        "MedInc": 1,
        "HouseAge": 1,
        "AveRooms": 3,
        "AveBedrms": 3,
        "Population": 3,
        "AveOccup": 5,
        "Latitude": 1,
        "Longitude": 1,
    }
    response = client.post(
        "/lab/predict",
        json=data,
    )
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_matches_output_model_expectation(client):
    data = {
        "MedInc": 1,
        "HouseAge": 1,
        "AveRooms": 3,
        "AveBedrms": 3,
        "Population": 3,
        "AveOccup": 5,
        "Latitude": 1,
        "Longitude": 1,
    }
    response = client.post(
        "/lab/predict",
        json=data,
    )
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)
    assert len(response.json().keys()) == 1


# Can I change the order of the message sent to the API?
# Python 3.7+ all dicts are ordered
# If we shuffle the keys then we have a new dict with the same data and should get the same prediction
def test_predict_order(client):
    base_message = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": 3,
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": 7,
        "Longitude": 8,
    }
    keys = list(base_message)
    # shuffle with a seed
    random.Random(42).shuffle(keys)

    # create new dictionary
    shuffled_message = {}
    for key in keys:
        shuffled_message[key] = base_message[key]

    # make sure the messages are not the same
    assert shuffled_message.keys != base_message.keys

    response_base = client.post(
        "/lab/predict",
        json=base_message,
    )
    response_shuffled = client.post(
        "/lab/predict",
        json=shuffled_message,
    )
    # compare predictions
    assert response_base.json()["prediction"] == response_shuffled.json()["prediction"]


# Add an extraneous feature
# Since we used pydantic.Extra.forbid this will return a 422 value_error.extra
def test_predict_extra_feature(client):
    data = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": 3,
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": 7,
        "Longitude": 8,
        "ExtraFeature": -1,
    }

    response = client.post(
        "/lab/predict",
        json=data,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "type": "extra_forbidden",
            "loc": ["body", "ExtraFeature"],
            "msg": "Extra inputs are not permitted",
            "input": -1,
        }
    ]


# Remove a feature
# pydantic should error since we're missing values
# This means our imputer actually never does anything
def test_predict_missing_feature(client):
    data = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": 3,
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": 7,
    }

    response = client.post(
        "/lab/predict",
        json=data,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "type": "missing",
            "loc": ["body", "Longitude"],
            "msg": "Field required",
            "input": {
                "MedInc": 1,
                "HouseAge": 2,
                "AveRooms": 3,
                "AveBedrms": 4,
                "Population": 5,
                "AveOccup": 6,
                "Latitude": 7,
            },
        }
    ]


# When we send both extra and missing features what happens?
# We get a message for each field that fails validation and have a list of errors returned
def test_predict_missing_and_extra_feature(client):
    data = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": 3,
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": 7,
        "ExtraFeature": 9,
    }

    response = client.post(
        "/lab/predict",
        json=data,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "type": "missing",
            "loc": ["body", "Longitude"],
            "msg": "Field required",
            "input": {
                "MedInc": 1,
                "HouseAge": 2,
                "AveRooms": 3,
                "AveBedrms": 4,
                "Population": 5,
                "AveOccup": 6,
                "Latitude": 7,
                "ExtraFeature": 9,
            },
        },
        {
            "type": "extra_forbidden",
            "loc": ["body", "ExtraFeature"],
            "msg": "Extra inputs are not permitted",
            "input": 9,
        },
    ]


# If we send in a bad type do we fail validation?
# here we see a string should have been a float
def test_predict_bad_type(client):
    data = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": "I am wrong",
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": 7,
        "Longitude": 8,
    }

    response = client.post(
        "/lab/predict",
        json=data,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "type": "float_parsing",
            "loc": ["body", "AveRooms"],
            "msg": "Input should be a valid number, unable to parse string as a number",
            "input": "I am wrong",
        }
    ]


# If we send a string value that can be coersed we should be fine
# the network is sending over the message as a string which gets parsed
# So everything is a string at some point but is validated on data instantiation
# This is called deserialization
def test_predict_bad_type_only_in_format(client):
    data = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": "3",
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": 7,
        "Longitude": 8,
    }

    response = client.post(
        "/lab/predict",
        json=data,
    )

    print(response.json())
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_bad_latitude_positive(client):
    data = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": 1,
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": 91,
        "Longitude": 8,
    }

    response = client.post(
        "/lab/predict",
        json=data,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "ctx": {"error": {}},
            "type": "value_error",
            "loc": ["body", "Latitude"],
            "msg": "Value error, Invalid value for Latitude",
            "input": 91,
        }
    ]


def test_predict_bad_latitude_negative(client):
    data = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": 1,
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": -91,
        "Longitude": 8,
    }

    response = client.post(
        "/lab/predict",
        json=data,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "ctx": {"error": {}},
            "type": "value_error",
            "loc": ["body", "Latitude"],
            "msg": "Value error, Invalid value for Latitude",
            "input": -91,
        }
    ]


def test_predict_bad_longitude_negative(client):
    data = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": 1,
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": 2,
        "Longitude": -181,
    }

    response = client.post(
        "/lab/predict",
        json=data,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "ctx": {"error": {}},
            "type": "value_error",
            "loc": ["body", "Longitude"],
            "msg": "Value error, Invalid value for Longitude",
            "input": -181,
        }
    ]


def test_predict_bad_longitude_positive(client):
    data = {
        "MedInc": 1,
        "HouseAge": 2,
        "AveRooms": 1,
        "AveBedrms": 4,
        "Population": 5,
        "AveOccup": 6,
        "Latitude": 0,
        "Longitude": 181,
    }

    response = client.post(
        "/lab/predict",
        json=data,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == [
        {
            "ctx": {"error": {}},
            "type": "value_error",
            "loc": ["body", "Longitude"],
            "msg": "Value error, Invalid value for Longitude",
            "input": 181,
        }
    ]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_output_model():
    with TestClient(app) as lifespanned_client:
      # Doing things the app model way
      # Get all routes from the app
      routes = app.routes

      # Extract the predict endpoint
      predict_route = None
      for in_route in routes:
          if in_route.path == "/lab/predict":
              predict_route = in_route
              break
      else:
          return False

      data = {
          "MedInc": 1,
          "HouseAge": 2,
          "AveRooms": 3,
          "AveBedrms": 4,
          "Population": 5,
          "AveOccup": 6,
          "Latitude": 7,
          "Longitude": 8,
      }

      # ensure there is a response model
      assert predict_route.response_model is not None
      # create an input model from the annotated values that state that there is an input model
      annotations = list(predict_route.endpoint.__annotations__.items())
      # Ensure that there is an input model defined, return hint optional
      assert len(annotations) >= 1

      input_model = list(filter(lambda x: x[0] != "return", annotations))
      # take the dict data and turn it into a model
      input_value = input_model[0][1].model_validate(data)

      # Test both async or regular code
      try:
          function_response = await predict_route.endpoint(input_value)
      except TypeError:
          function_response = predict_route.endpoint(input_value)

      # Verify that the returned output is actually a class and not just a hardcoded dictionary
      assert isinstance(function_response, predict_route.response_model)


"""
@pytest.mark.asyncio
def test_multi_predict():
    with TestClient(app) as lifespanned_client:
    # Sample data for multiple houses
        data = {
          "houses": [
            {
                "MedInc": 2.0, "HouseAge": 20.0, "AveRooms": 4.0, "AveBedrms": 1.0, 
                "Population": 800, "AveOccup": 3.0, "Latitude": 37.88, "Longitude": -122.23
            },
            {
                "MedInc": 4.0, "HouseAge": 30.0, "AveRooms": 6.0, "AveBedrms": 2.0, 
                "Population": 1200, "AveOccup": 2.5, "Latitude": 34.05, "Longitude": -118.24
            }
          ]
       }

       response = lifespanned_client.post("/lab/bulk-predict", json=data)
       assert response.status_code == 200
    
       # Assert that the response contains a list of predictions
       print (response.json())
       predictions = response.json()["predictions"]
       assert isinstance(predictions, list)
       assert len(predictions) == 2  # Since we sent 2 houses
    

    if __name__ == "__main__":
      pytest.main()
"""
def test_multi_predict(client):
    data = {
      "houses": [
        {
            "MedInc": 2.0, "HouseAge": 20.0, "AveRooms": 4.0, "AveBedrms": 1.0, 
            "Population": 800, "AveOccup": 3.0, "Latitude": 37.88, "Longitude": -122.23
        },
        {
            "MedInc": 4.0, "HouseAge": 30.0, "AveRooms": 6.0, "AveBedrms": 2.0, 
            "Population": 1200, "AveOccup": 2.5, "Latitude": 34.05, "Longitude": -118.24
        }
          ]
    }

    response = client.post("/lab/bulk-predict", json=data)
    assert response.status_code == 200

    # Assert that the response contains a list of predictions
    print(response.json())
    predictions = response.json()["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 2  # Since we sent 2 houses


    if __name__ == "__main__":
      pytest.main()


def test_multi_predict_cache(client):
    data = {
      "houses": [
        {
            "MedInc": 2.0, "HouseAge": 20.0, "AveRooms": 4.0, "AveBedrms": 1.0, 
            "Population": 800, "AveOccup": 3.0, "Latitude": 37.88, "Longitude": -122.23
        },
        {
            "MedInc": 4.0, "HouseAge": 30.0, "AveRooms": 6.0, "AveBedrms": 2.0, 
            "Population": 1200, "AveOccup": 2.5, "Latitude": 34.05, "Longitude": -118.24
        }
          ]
    }

    response1 = client.post("/lab/bulk-predict", json=data)
    assert response1.status_code == 200

    response2 = client.post("/lab/bulk-predict", json=data)
    assert response2.status_code == 200

    if __name__ == "__main__":
      pytest.main()

    

def test_health(client):
    response = client.get("/lab/health")
    assert response.status_code == 200
