"""
Plot one week of events loaded from file (starting from the earliest event).

Examples:
    plot_events.py --from events.json
    
Usage:
    plot_events.py [--from=<FILE>]

Options:
    -h --help         Show this screen.
    -f --from=<FILE>  File containing a list of event descriptions [default: default]
"""
from os.path import join, dirname, exists
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import json
import pytz
import matplotlib.ticker as ticker
import re

from datetime import datetime, time, timedelta
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from docopt import docopt
from datetime import time
from math import ceil
from collections import namedtuple
from typing import List, Dict
from schedulingassistant.data import Event
from schedulingassistant.conversion_utils import time_to_float


# TODO:
# * Complete all docstrings for function in this file
# * Add tests for `str_to_datetime`


DAY_COUNT = 7


def main(sys_args: List[str] = []) -> None:
    args = parse_args(docopt(__doc__, argv=sys_args))

    try:
        events = extract_events_from(args['event_file'])
        plot_events(events)
        return 0
    except Exception as e_info:
        print(e_info)
        return 1


def parse_args(args: Dict[str, str]) -> Dict[str, any]:
    """Parse the arguments passed in, and return a dictionary containing the final input values for the application."""
    DEFAULT_EVENT_FILE = "generated_events.json"

    event_file_arg = args['--from']

    event_file_path = event_file_arg
    if event_file_arg == "default":
        event_file_path = join(dirname(__file__), DEFAULT_EVENT_FILE)

    if exists(event_file_path):
        return { 'event_file': event_file_path }
    else:
        raise ValueError("File '" + event_file_path + "' could not be found.")


def plot_events(events: List[Event]) -> None:
    fig = plt.figure(figsize=(10, 16))
    fig.tight_layout()
    plt.title('Events', y=1, fontsize=14)

    ax = fig.add_subplot(1, 1, 1)

    # X
    ax.set_xlim(0.5, DAY_COUNT + 0.5)
    earliest_date = min([e.start_datetime.date() for e in events])
    date_labels = [(earliest_date + timedelta(days=i)).strftime("%d/%m/%Y") for i in range(DAY_COUNT + 1)]
    ax.set_xticks(range(1, DAY_COUNT + 1))
    ax.set_xticklabels(date_labels)
    plt.tick_params(bottom=False) # Hide ticks

    # Y
    start_of_day = 0
    end_of_day = 24

    ax.set_ylim(end_of_day, start_of_day)

    block_times = np.arange(start_of_day, end_of_day, float(5.0/60.0))
    ax.set_yticks(block_times)

    hour_labels = [("{0}:00".format(int(b)) if b.is_integer() else "") for b in block_times]
    ax.set_yticklabels(hour_labels)

    # Create the horizontal timeblock grid lines
    ax.grid(axis='y', linestyle='-', linewidth=0.3, color="black", alpha=0.05)

    grid_lines = ax.yaxis.get_gridlines()
    for h in range(end_of_day):
        label = "{0}:00".format(h)
        label_idx = hour_labels.index(label)
        grid_lines[label_idx].set_alpha(1.0)

    # Go through and make all hour grid lines bold 
    # https://stackoverflow.com/questions/53781180/polar-plot-put-one-grid-line-in-bold

    # Plot the events
    for e in events:
        plot_event(e, earliest_date, ax)

    # Save this output to an image file and open it
    img_name = 'events.png'
    plt.savefig(img_name, dpi=400, bbox_inches='tight')
    img = Image.open(img_name)
    img.show()


def extract_events_from(events_file_path: str) -> List[Event]:
    """todo, also add docstrings to all other functions in this file"""
    events = []

    with open(events_file_path) as events_file:
        json_events = json.load(events_file)

        for e in json_events:
            name = e['name']
            start = str_to_datetime(e['start_datetime'])
            end = str_to_datetime(e['end_datetime'])

            events.append(Event(name, start, end))

    return events


def str_to_datetime(input_str: str) -> datetime:
    """Parse a string `input_str` and return a corresponding `datetime` object."""
    microseconds = 0

    if '.' in input_str:
        seconds_decimal_component_match = re.match(r"[^.]*\d+[^.]*(.\d+)", input_str)
        if seconds_decimal_component_match:
            decimal_component_str = seconds_decimal_component_match.group(1)
            input_str = input_str.replace(decimal_component_str, '')
            microseconds = int(float("0" + decimal_component_str) * 1000000)

    output = datetime.strptime(input_str, "%Y-%m-%dT%H:%M:%S%z").replace(microsecond=microseconds, tzinfo=pytz.utc)

    return output


def plot_event(e: Event, earliest_date: datetime.date, ax) -> None:
    boxes = convert_event_to_boxes(e)
    
    # An index representing the first day that the start of this event should be on
    day_offset = (e.start_datetime.date() - earliest_date).days

    color = rand_hex_col()
    start_hour = e.start_datetime.hour
    start_min = e.start_datetime.minute
    event_label = '{0}:{1:0>2} {2}'.format(start_hour, start_min, e.name) 

    for box_idx in range(len(boxes)):
        label = event_label if box_idx == 0 else ""
        draw_event_box(boxes[box_idx], day_offset, label, color, ax)


# An `EventBox` represents a window of time that can be drawn with one rectangle on a calendar with multiple days in
# different columns. E.g 9am - 10am would be valid `EventBox`, but 11pm - 1am would not as this would have to be broken
# down into two windows.
EventBox = namedtuple('EventBox', ['column_idx', 'start_time_float', 'end_time_float'])


def draw_event_box(box: EventBox, day_offset: int, label: str, color: str, ax):
    """Draws an event box on the plot using a day index (used internally to calculate the horizontal components of the
    box, and two start/end floats representing percentages through the day, used to calculate the vertical components."""
    top = box.start_time_float
    bottom = box.end_time_float
    left = 0.5 + box.column_idx + day_offset

    # If this event would be drawn outside the view of the plot
    if left >= 7.0:
        return

    padding_between_days = 0.05
    right = left + 1 - padding_between_days

    # Draw boxes and labels on top of everything else
    z = 2.0

    box = FancyBboxPatch(
        (left, top),
        abs(right - left),
        abs(bottom - top),
        boxstyle="round,pad=-0.0040,rounding_size=0.02",
        ec="black",
        fc=color,
        lw=0.2,
        zorder=z,
        mutation_aspect=1)
    
    ax.add_patch(box)

    plt.text(left + 0.01, top + 0.01, label, va='top', fontsize=3, zorder=z)


def convert_event_to_boxes(event: Event) -> List[EventBox]:
    """Takes in an event and converts this into a list of boxes that when combined completely cover the time allocated
    to this event. Usually, this list will contain a single EventBox as many events start and end on the same day, but
    any events split across multiple day boundaries will be split into multiple boxes."""
    start_date = event.start_datetime.date()
    end_date = event.end_datetime.date()

    start_time_float = time_to_float(event.start_datetime.time())
    end_time_float = time_to_float(event.end_datetime.time())
    days_spanned = (end_date - start_date).days + 1

    boxes = []

    if days_spanned == 1:
        boxes.append(EventBox(0, start_time_float, end_time_float))
    else:
        boxes.append(EventBox(0, start_time_float, 24.0))
        
        for i in range(max(0, days_spanned - 2)):
            boxes.append(EventBox(i + 1, 0.0, 24.0))

        boxes.append(EventBox(days_spanned - 1, 0.0, end_time_float))

    return boxes


# Create rounded box for the event with a random colour
# https://stackoverflow.com/questions/58425392/bar-chart-with-rounded-corners-in-matplotlib
def rand_hex_col() -> str:
    r = lambda: 128 + random.randint(0, 127)
    return '#%02X%02X%02X' % (r(),r(),r())


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
