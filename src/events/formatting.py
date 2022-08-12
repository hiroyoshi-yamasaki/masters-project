import logging
from math import isnan
from pathlib import Path
import re
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(file_name="formatting")
logger.setLevel(logging.INFO)

id_to_name = {1: "word",
              2: "word",
              3: "word",
              4: "word",
              5: "word",
              6: "word",
              7: "word",
              8: "word"}

# These Events are not used in this project
#             10: "block",
#             15: "offset",
#             20: "fixation",
#             30: "pause",
#             40: "question",
#             Re-assigned IDs (these are assigned 1-3 which overlaps with word conditions)
#             50: "response/1",
#             60: "response/2",
#             70: "response/3"}

########################################################################################################################
# STIMULI PROCESSING                                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #
# Processing related to stimuli prior to running the dataset generation pipeline. Can be run independently of the      #
# said pipeline. It can also be combined with manual correction.                                                       #
# `format_event_data` can be used to convert `event.tsv` into cleaner format                                           #
########################################################################################################################


def format_event_data(events_path, stimuli_path) -> [pd.DataFrame, list]:
    """
    Clean up events data by removing the unnecessary components and reformatting the data
    :param events_path: path to the CSV file provided by the original dataset
    :param stimuli_path: path to the stimuli.txt file
    :return:
        events_df: pd.DataFrame, cleaned dataframe object with following columns
            `sample`: sampling index, int
            `type`: either `word`, `block` or `empty`. Only `word` matters to this project
            `onset`: onset time in seconds
            `form`: actual word form presented, e.g. `zijn`
            `sentence`: sentence ID assigned to the stimulus
            `position`: position of the word within the sentence
            `ID`: word ID
        rejected_list: list of events rejected due to bad formats
    """

    events_df = pd.read_csv(events_path, sep="\t")
    with open(stimuli_path, "r") as f:
        stimuli_text = f.read()

    _, position_to_id = _assign_ids(stimuli_text)

    events_df = _clean_df(events_df)
    events_df, rejected_list = _add_sentence_column(events_df, stimuli_text)
    events_df.dropna(axis=0, inplace=True)
    events_df = _add_ids(events_df, position_to_id)
    return events_df, rejected_list


def _assign_ids(text: str):
    """
    From stimuli.txt content construct two dictionaries by assigning and ID to every token in the text
    :param text: text file of stimuli.txt. Contains list of stimuli in the format: [stimulus no.] [sentence/list]\n
    :return:
        id_to_word: dictionary, key = token ID, value = token string
        position_to_id: dictionary, key = sentence number, value = dictionary {position in sentence: token ID}
    """

    ids = 0
    id_to_word = {}
    position_to_id = {}

    for line in text.splitlines():
        sentence_id = int(re.findall(r"^\d+(?=\s)", line)[0])
        line = re.sub(r"\d+", "", line)  # remove the sentence number
        position_to_id[sentence_id] = {}

        words = line.split()
        for pos, word in enumerate(words):
            id_to_word[ids] = word
            position_to_id[sentence_id][pos] = ids
            ids += 1

    return id_to_word, position_to_id


def _clean_df(df):
    """
    Select relevant information and discard the rest. There are unnecessary events such as `frontpanel trigger` or
    `trial` all summarised in the column `type`. Moreover, there are multiple rows corresponding to a single event.
        E.g. sample=3319 has row with value=frontpanel trigger, value=UPPT001, value=Picture
    To solve this, group events with same sampling index and assign single type while discarding unnecessary info
    :param df: original dataframe with columns:
        `onset`: onset time in seconds
        `duration`: discarded
        `sample`: sampling index
        `type`: original type description. To be edited later
        `value`: original value of the event. To be edited later
    :return: cleaned dataframe with columns:
        `sample`: sampling index
        `type`: simplified `type` with values
            `fixation`, `response/1`, `response/2`, `response/3`, `block`, `question`, `word`, `empty`, `block`
        `onset`: onset time in seconds
        `form`: word form
    """

    sample_list, type_list, onset_list, form_list = [], [], [], []

    curr_sample = 1
    curr_event = []
    for idx, row in df.iterrows():

        # Group events with the same sampling index
        if row["sample"] in [curr_sample, curr_sample + 1]:  # sample value can vary by max. 1 sample
            curr_event.append(row)

        else:  # new event
            _match_event(curr_event, sample_list, type_list, onset_list, form_list)
            curr_sample = row["sample"]
            curr_event = [row]

    df = pd.DataFrame({"sample": sample_list, "type": type_list, "onset": onset_list, "form": form_list})
    return df


def _match_event(events, sample_list, type_list, onset_list, form_list):
    """
    Match relevant information and append to list
    :param events: list of rows in the event (Series)
    :param sample_list: list of sample values
    :param type_list: list of type values
    :param onset_list: list of onset values
    :param form_list: list of form values
    """

    matched = False
    event_name = None
    form = None

    for row in events:

        # Response
        if row["type"] == "Response":

            event_name = "response/"

            if row["value"] == "1":
                event_name += "1"
            elif row["value"] == "2":
                event_name += "2"
            elif row["value"] == "3":
                event_name += "3"

            matched = True
            break

        elif row["type"] in ["trial", "UDIO001"]:  # ignore these
            break

        elif row["type"] == "Picture":
            if re.match(r"(ZINNEN|WOORDEN).*", row["value"]):
                event_name = "block"
                matched = True
                form = row["value"]
                break

            elif row["value"].startswith("FIX"):
                event_name = "fixation"
                matched = True
                break

            elif row["value"].startswith("QUESTION"):
                event_name = "question"
                matched = True
                break

            elif row["value"] in ["blank", "pause", "ISI", "PULSE MODE 0", "PULSE MODE 1", "PULSE MODE 2",
                                  "PULSE MODE 3", "PULSE MODE 4", "PULSE MODE 5"]:  # ignore
                break

            # Word
            elif re.match(r"^\d\s?[a-zA-Z]+", row["value"]):
                event_name = "word"
                form = row["value"]
                form = re.sub(r"\d|\s|\.", "", form)  # remove space and numbers
                matched = True
                break

            elif re.match(r"^\d+\s+\d+", row["value"]):  # e.g. 5 300
                event_name = "empty"
                form = ""
                matched = True
                break

            else:
                t, v = row["type"], row["value"]
                raise ValueError(f"The row with type '{t}' and value '{v}' was not matched")

    if matched:
        sample_list.append(events[0]["sample"])
        type_list.append(event_name)
        onset_list.append(events[0]["onset"])
        form_list.append(form)


def _add_sentence_column(df: pd.DataFrame, stimuli_text: str):
    """
    todo comment
    :param df:
    :param stimuli_text:
    :return:
    """

    df["sentence"] = ""
    df["position"] = ""

    word_list = None
    rejected_list = []

    for idx, row in df.iterrows():

        if row["type"] == "fixation":  # new sentence is preceded by "fixation"

            df = _modify_df(df, word_list, stimuli_text, rejected_list)

            word_list = []

        elif row["type"] == "word":
            word_list.append((idx, row["form"]))  # remember index: word matching

    df = _modify_df(df, word_list, stimuli_text, rejected_list)
    return df, rejected_list


def _modify_df(df: pd.DataFrame, word_list, stimuli_text, rejected_list):
    # todo comment

    if word_list is not None:

        # Find the sentence that matches the list of words given here
        sentence_number = _find_sentence(word_list, stimuli_text)
        if sentence_number is None:
            words = [y for x, y in word_list]  # word_list is a list of tuples
            rejected = " ".join(words)
            rejected_list.append(rejected)
            logger.debug(f"The sentence '{rejected}' was rejected")

        # Modify the DataFrame
        for s_idx, (w_idx, word) in enumerate(word_list):
            df.at[w_idx, "sentence"] = sentence_number
            df.at[w_idx, "position"] = s_idx  # position with the sentence
    return df


def _find_sentence(word_list, stimuli_text):
    # todo comment

    stimuli_list = stimuli_text.splitlines()
    for stimulus in stimuli_list:
        stimulus_words = stimulus.split()
        sentence_number = int(stimulus_words[0])  # first element is the sentence number
        stimulus_words = stimulus_words[1:]       # the rest are the real sentences

        # Assume a match until it fails
        found = True
        min_length = min(len(word_list), len(stimulus_words))  # to avoid out of range error
        for idx in range(min_length):
            if stimulus_words[idx].lower() != word_list[idx][1].lower():  # word_list is a list of tuples (index, word)
                found = False
                break

        if found:
            return sentence_number

    return None


def _add_ids(df: pd.DataFrame, position_to_id: dict):
    # todo comment

    df["ID"] = ""

    for idx, row in df.iterrows():
        sentence_number = row["sentence"]
        position = row["position"]
        if sentence_number is not None and position != "":  # missing sentence or position
            if sentence_number in position_to_id:
                if position in position_to_id[sentence_number]:
                    token_id = position_to_id[sentence_number][position]
                    df.at[idx, "ID"] = token_id

    return df


########################################################################################################################
# EVENT VALIDATION                                                                                                     #
# -------------------------------------------------------------------------------------------------------------------- #
# Validating the events array generated by MNE signals. Reject any events that are inconsistent                        #
########################################################################################################################


def get_event_array(events: np.array, event_path: Path, dictionary_path: Path, simplify_mode: str,
                    strict=False, threshold=30) -> np.array:
    """
    Compare MNE events array with dataframe from .csv files. Drop any inconsistent events.
    :param events: events array generated with mne.find_events
    :param event_path: path to .csv file
    :param dictionary_path: path to .csv file containing POS information
    :param simplify_mode: whether to use token IDs (`index`) or noun (0) vs. verb (1) comparison (`binary`) for event
        values
    :param strict: todo whats strict
    :param threshold: todo whats threshold
    :return: validated events array
    """

    logger.info("Verifying the event array against dataframe data")

    original_events = events[0]  # at original sfreq
    new_events = events[1]       # downsampled
    df = pd.read_csv(str(event_path))

    valid_events = []
    invalid_events = []
    for idx, o_event in enumerate(original_events):

        # Ignore events that are not in the dictionary
        if o_event[2] in id_to_name:
            mne_event = id_to_name[o_event[2]]
        else:
            continue

        # Sample value can vary by maximum on 1 on either side
        if o_event[0] in df["sample"].values:
            df_event = df.loc[df["sample"] == o_event[0]]
        elif o_event[0] - 1 in df["sample"].values:
            df_event = df.loc[df["sample"] == o_event[0] - 1]
        elif o_event[0] + 1 in df["sample"].values:
            df_event = df.loc[df["sample"] == o_event[0] + 1]
        elif not strict:
            below_threshold, df_event = _check_difference(o_event, df, threshold=threshold)
            if not below_threshold:
                invalid_events.append(o_event)
                continue
        else:
            invalid_events.append(o_event)
            continue

        # The event name is identical
        if df_event["type"].values.size == 0:
            print()
        if df_event["type"].values[0] == mne_event:

            if not isnan(df_event["ID"].values[0]):
                # Replace MNE ID with Token ID
                new_events[idx, 2] = int(df_event["ID"].values[0])
                valid_events.append(new_events[idx])

        # These are neither valid nor relevant
        elif mne_event == "word" and df_event["type"].values[0] == "empty":  # blank "words", e.g. 5 300
            pass
        else:
            invalid_events.append(o_event)

    logger.info(f"{len(valid_events) / (len(valid_events) + len(invalid_events)) * 100}% valid")
    logger.info(f"Total of {len(valid_events)} events added")

    events = np.array(valid_events)
    events = _simplify(events, dictionary_path, mode=simplify_mode)
    return events


def _check_difference(o_event, df, threshold):
    """
    todo: comment
    :param o_event:
    :param df:
    :param threshold:
    :return:
    """

    if len(df.loc[abs(df["sample"] - o_event[0]) < threshold]) > 0:  # if minimum error less than threshold
        return True, df.loc[abs(df["sample"] - o_event[0]) < threshold]
    else:
        return False, None


def _simplify(events: np.array, df_path: Path, mode="index"):
    """
    Drop any events not noun or verb
    :param events: events array
    :param df_path: path to .csv file containing POS information
    :param mode: `index` uses the token ID, `binary` uses `0` (noun) vs `1` (verb)
    :return: simplified events array
    """

    df = pd.read_csv(str(df_path))
    df["POS"] = df["POS"].apply(lambda x: 0 if x == "N" else 1)

    event_list = []
    for event in events:
        if event[2] in df["Token ID"].values:  # only keep nouns or verbs
            if mode == "binary":
                event[2] = df[df["Token ID"] == event[2]]["POS"]  # convert to noun vs verb binary
            event_list.append(event)
    events = np.array(event_list)
    return events


def select_conditions(events: np.array, mode="both") -> np.array:
    """
    Selects events based on the condition. There are sentence and word list conditions
    :param events: events array
    :param mode: Options. Valid are `both`, `sentence` and `list`
    :return: events with only selected event type
    """

    logger.debug(f"Selecting {mode}")

    if mode == "sentence":
        events = events[np.where(events[:, 2] < 4598)]  # 4597 = largest sentence token ID
    elif mode == "list":
        events = events[np.where(events[:2] >= 4598)]
    return events


if __name__ == "__main__":
    stimuli = Path("/Users/hiro/Downloads/stimuli.txt")
    events = Path("/Users/hiro/Downloads/sub-V1010_task-visual_events - sub-V1010_task-visual_events.tsv")

    df, rejects = format_event_data(events_path=events, stimuli_path=stimuli)
    print(df)
    pass
