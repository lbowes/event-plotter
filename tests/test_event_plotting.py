from plot_events import *
from helper_functions import *
from random import randint


# TODO: Consider separating out all conversion functions into their own file

def test_convert_event_to_boxes(faker) -> None:
    """Test that an `Event` can be converted into a list of `EventBox`s on different days (one event can be split across
    multiple days, and when plotted on a calendar, this event must be drawn with multiple boxes)."""
    # Case when an event starts and finished on the same day
    start_datetime = datetime(2020, 11, 5, 9, 0)
    end_datetime = datetime(2020, 11, 5, 14, 0)
    test_event = Event(faker.word(), start_datetime, end_datetime)

    expected_output = [
        EventBox(0, time_to_float(start_datetime.time()), time_to_float(end_datetime.time()))
    ]

    assert convert_event_to_boxes(test_event) == expected_output

    # Case when an event overlaps one day boundary
    start_datetime = datetime(2020, 11, 5, 18, 10)
    end_datetime = datetime(2020, 11, 6, 9, 0)
    test_event = Event(faker.word(), start_datetime, end_datetime)

    expected_output = [
        EventBox(0, time_to_float(start_datetime.time()), 24.0),
        EventBox(1, 0.0, time_to_float(end_datetime.time()))
    ]

    assert convert_event_to_boxes(test_event) == expected_output

    # Case when an event overlaps two day boundaries
    start_datetime = datetime(2020, 11, 5, 18, 10)
    end_datetime = datetime(2020, 11, 7, 9, 0)
    test_event = Event(faker.word(), start_datetime, end_datetime)

    expected_output = [
        EventBox(0, time_to_float(start_datetime.time()), 24.0), # the first day
        EventBox(1, 0.0, 24.0), # second day 
        EventBox(2, 0.0, time_to_float(end_datetime.time())) # third day
    ]

    assert convert_event_to_boxes(test_event) == expected_output
