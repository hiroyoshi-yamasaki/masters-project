import logging

from pathlib import Path

import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(file_name="conditions")
logger.setLevel(logging.INFO)


########################################################################################################################
# CONDITION PROCESSING                                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
# Utilities for filtering relevant experimental conditions and converting them into appropriate format such that it    #
# can be used as labels for Machine Learning algorithms                                                                #
########################################################################################################################

# Convert generic y to specific y based on conditions given ############################################################


def convert_y(y: np.array, mode: str, df_dir: Path, to_index: bool, balance: bool, params: dict):
    """
    Main function for conversion. Calls appropriate conversion function based on the mode given
    :param y: generic y of the shape (n_events,)
    :param mode: Type of conversion
        'nv': nouns vs. verbs
        'length': short vs. long words
        'frequency': frequent vs. rare
        'tense': present vs. past
        'person': 1 vs. 2 vs 3
        'v-number': sg. vs. pl. for verbs
        'voice': active vs. passive
        'gender': m. vs. n. vs. f.
        'n-number': sg. vs. p. for nouns
    :param df_dir: directory containing relevant .csv files
    :param to_index: bool, if true, converts to an integer instead of the name of the condition
    :param balance: balance number of items in classes by cropping randomly
    :param params: parameters specific to conditions
    :return:
        y: converted y, (2, n_events) first row corresponds to the class, second to indices
        included: array of indices corresponding to included events. For indexing corresponding x
    """

    logger.info(f"Converting the event labels using the mode {mode}")

    if mode == "nv":
        id_to_cond = _to_nv(df_dir=df_dir, to_index=to_index, params=params)

    elif mode == "length":
        id_to_cond = _to_length(df_dir=df_dir, to_index=to_index, params=params)

    elif mode == "frequency":
        id_to_cond = _to_frequency(df_dir=df_dir, to_index=to_index, params=params)

    elif mode == "tense":
        id_to_cond = _to_tense(df_dir=df_dir, to_index=to_index, params=params)

    elif mode == "person":
        id_to_cond = _to_person(df_dir=df_dir, to_index=to_index, params=params)

    elif mode == "v-number":
        id_to_cond = _to_v_number(df_dir=df_dir, to_index=to_index, params=params)

    elif mode == "voice":
        id_to_cond = _to_voice(df_dir=df_dir, to_index=to_index)

    elif mode == "gender":
        id_to_cond = _to_gender(df_dir=df_dir, to_index=to_index, params=params)

    elif mode == "n-number":
        id_to_cond = _to_n_number(df_dir=df_dir, to_index=to_index)

    else:
        raise ValueError(f"Unknown mode \'{mode}\'")

    y, included = _to_arrays(y, id_to_cond)

    # Balance the number of items per class
    if balance:
        y, idx = _balance_classes(y)
        included = included[idx]

    return y, included


def _to_dict(df: pd.DataFrame, key: str, value: str, mapper: dict, to_index: bool):
    """
    Generate a dictionary from token ID to condition from dataframe
    :param df: dataframe with relevant information
    :param key: column with values which becomes the key for the dictionary
    :param value: column with values becomes the value for the dictionary
    :param mapper: dictionary which pairs condition name and value, e.g. 'N': 0
    :param to_index: if true, convert to numbers rather than name.
    :return:
        id_to_cond: dictionary mapping from ID to condition
    """

    id_to_cond = {}

    for idx, row in df.iterrows():
        if to_index:
            id_to_cond[row[key]] = mapper[row[value]]
        else:
            id_to_cond[row[key]] = row[value]
    return id_to_cond


def _to_arrays(y: np.array, id_to_cond: dict):
    """
    Replace y with appropriate conditions
    :param y: original y
    :param id_to_cond: dictionary mapping from token ID to condition, Token ID: condition ID
    :return:
        y: (2, n_events), first row corresponds to condition ID,
        second row contains original indices (for recovering wrong items later)
        included: indices for events included
    """

    conditions = []
    tokens = []
    included = []

    for idx, item in enumerate(y):
        if item in id_to_cond:
            conditions.append(id_to_cond[item])
            tokens.append(item)
            included.append(idx)

    return np.array([conditions, tokens]), np.array(included)


def _balance_classes(y: np.array):
    """
    Balance the number of items per class by dropping randomly.
    :param y: array to be modified, (2, n_events)
    :return:
        y: modified y
        idx: index used to modify (needed for cropping `included`)
    """

    classes = list(set(y[0].tolist()))

    # Get the least frequent class
    counts = np.bincount(y[0])
    least_class = counts.argmin()
    size = counts[least_class]

    # Randomly sample with same size
    idx_list = []
    for cl in classes:
        idx = np.where(y[0] == cl)
        idx = np.random.choice(idx[0], size, replace=False)
        idx_list.append(idx)

    idx = np.concatenate(idx_list)
    y = y[:, idx]
    return y, idx


# Nouns vs. Verbs ######################################################################################################


def _to_nv(df_dir: Path, to_index: bool, params: dict):
    """
    Noun vs. Verb
    :param df_dir: directory with .csv files
    :param to_index: if true, convert to indices rather than names
    :param params:
        number: specify number
        tense: specify tense
        person: specify person
        voice: specify voice
        allow_non_finite: if true, include non-finite words
        allow_complex: if true, include multiple word verbs
        allow_aux: if true, include auxiliary verbs
        allow_ambiguous_gender: if true, include ambiguous gender e.g. m./f.
        allow_common_gender: if true, include common gender nouns
        allow_diminutives: if true, include diminutive words
        allow_uncountables: if true, include uncountalbe nouns
        allow_proper: if true, include proper nouns
    :return:
        dictionary
    """

    nv_df = pd.read_csv(df_dir / "NV.csv")
    v_df = pd.read_csv(df_dir / "Verbs-Grammatical.csv")
    n_df = pd.read_csv(df_dir / "Nouns-Grammatical.csv")

    # Filter verbs
    v_df = _select_verbs(v_df, number=params["number"], tense=params["tense"], person=params["person"],
                         voice=params["voice"], allow_non_finite=params["finite"], allow_complex=params["complex"],
                         allow_aux=params["aux"])

    # Filter nouns
    n_df = _select_nouns(n_df, allow_ambiguous_gender=params["ambiguous"], allow_common_gender=params["common"],
                         allow_diminutives=params["diminutive"], allow_uncountables=params["uncountable"],
                         allow_proper=params["proper"])

    # Remove items not found in the verb and noun dataframes
    df_v = pd.merge(nv_df, v_df, how="left", on="Token ID").dropna(axis=0)[["Token ID", "POS"]]
    df_n = pd.merge(nv_df, n_df, how="left", on="Token ID").dropna(axis=0)[["Token ID", "POS"]]
    df = pd.concat([df_v, df_n])

    return _to_dict(df=df, key="Token ID", value="POS", mapper={"N": 0, "V": 1}, to_index=to_index)


def _select_verbs(verbs: pd.DataFrame, number=None, tense=None, person=None, voice=None,
                  allow_non_finite=False, allow_complex=False, allow_aux=False):
    """
    Filter out verbs
    :param verbs: dataframe with relevant information
    :param number: specify number
    :param tense: specify tense
    :param person: specify person
    :param voice: specify voice
    :param allow_non_finite: if true, include non-finite words
    :param allow_complex: if true, include multiple word verbs
    :param allow_aux: if true, include auxiliary verbs
    :return:
        cropped dataframe
    """

    # Filter by number
    if number is not None:
        if number not in ["sg.", "pl."]:
            verbs["Number"] = verbs[verbs["Number"].isin(["sg.", "pl."])]

        verbs = verbs[verbs["Number"] == number]

    # Filter by tense
    if tense is not None:
        if tense not in ["present", "past"]:
            verbs = verbs[verbs["Tense"].isin(["present", "past"])]

        verbs = verbs[verbs["Tense"] == tense]

    # Filter by person
    if person is not None:
        if person not in [1, 2, 3]:
            verbs = verbs[verbs["Person"].isin([1, 2, 3])]

        verbs = verbs[verbs["Person"] == person]

    # Filter by voice
    if voice is not None:
        if tense not in ["active", "passive"]:
            verbs = verbs[verbs["Voice"].isin(["active", "passive"])]

        verbs = verbs[verbs["Voice"] == voice]

    # Filter by finiteness
    if allow_non_finite:
        verbs = verbs[verbs["Finite"] == "finite"]

    # Filter by simple (single word) vs complex (multiple word)
    if not allow_complex:
        verbs = verbs[verbs["Complex"] is False]

    # Filter auxiliary words
    if not allow_aux:
        verbs = verbs[verbs["Complex"] != "auxiliary"]

    return verbs


def _select_nouns(nouns: pd.DataFrame, allow_ambiguous_gender=False, allow_common_gender=False,
                  allow_diminutives=False, allow_uncountables=False, allow_proper=False):
    """
    Select appropriate nouns
    :param nouns: dataframe with relevant information
    :param allow_ambiguous_gender: if true, include ambiguous gender e.g. m./f.
    :param allow_common_gender: if true, include common gender nouns
    :param allow_diminutives: if true, include diminutive words
    :param allow_uncountables: if true, include uncountalbe nouns
    :param allow_proper: if true, include proper nouns
    :return:
        dictionary
    """

    # Filter ambiguous genders
    if not allow_ambiguous_gender:
        nouns = nouns[~nouns["Gender"].isin(["n./c.", "m./f."])]

    # Filter common gender
    if not allow_common_gender:
        nouns = nouns[~nouns["Gender"].isin(["n./c.", "c."])]

    # Filter diminutives
    if not allow_diminutives:
        nouns = nouns[nouns["Diminutive"] is False]

    # Filter uncountable
    if not allow_uncountables:
        nouns = nouns[nouns["Number"] != "uncount."]

    # Filter proper nouns
    if not allow_proper:
        nouns = nouns[nouns["Proper"] is False]

    return nouns


########################################################################################################################
# Word length                                                                                                          #
########################################################################################################################


def _to_length(df_dir: Path, to_index: bool, params: dict):
    """
    Word length, long vs. short
    :param df_dir: directory with .csv files
    :param to_index: if true, convert to indices rather than names
    :param params:
        lower: boundary between short and medium
        upper: boundary between medium and long
        medium: if true, include medium and group in three groups (instead of two)
    :return:
        dictionary
    """

    df = pd.read_csv(df_dir / "NV.csv")

    df["Length"] = df["Word"].apply(lambda x: len(x))

    # Conditions for grouping into 3 groups
    conditions = [(df["Length"] < params["lower"]),
                  (params["lower"] <= df["Length"]) & (df["Length"] < params["upper"]),
                  (params["upper"] <= df["Length"])]
    df["Group"] = np.select(conditions, ["short", "medium", "long"])

    if not params["medium"]:
        df["Group"] = df[(df["Group"] == "short") | (df["Group"] == "long")]["Group"]
        mapper = {"short": 0, "long": 1}
    else:
        mapper = {"short": 0,  "medium": 1, "long": 2}

    df.dropna(axis=0, inplace=True)

    return _to_dict(df=df, key="Token ID", value="Group", mapper=mapper, to_index=to_index)


########################################################################################################################
# Word frequency                                                                                                       #
########################################################################################################################


def _to_frequency(df_dir: Path, to_index: bool, params: dict):
    """
    Frequency, frequent vs. rare
    :param df_dir: directory with .csv files
    :param to_index: if true, convert to indices rather than names
    :param params:
        mode: either `frequency` to use log frequency or `cd` to use contextual diversity
        lower: boundary between short and medium
        upper: boundary between medium and long
        medium: if true, include medium and group in three groups (instead of two)
    :return:
        dictionary
    """

    df = pd.read_csv(df_dir / "NV.csv")
    sub = pd.read_csv(df_dir / "SUBTLEX-NL.csv")
    df = pd.merge(df, sub, how="left", on="Word")
    df = df[["Token ID", "Frequency", "CD", "POS"]]

    if params["mode"] == "noun":
        df = df[df["POS"] == "N"]
    elif params["mode"] == "verb":
        df = df[df["POS"]]

    column = "Frequency" if params["mode"] == "frequency" else "CD"

    # Conditions for grouping into 3 groups
    conditions = [(df[column] < params["lower"]),
                  (params["lower"] <= df[column]) & (df["Length"] < params["upper"]),
                  (params["upper"] <= df[column])]
    df["Group"] = np.select(conditions, ["short", "medium", "long"])

    if not params["medium"]:
        df["Group"] = df[(df["Group"] == "short") | (df["Group"] == "long")]["Group"]
        mapper = {"short": 0, "long": 1}
    else:
        mapper = {"short": 0,  "medium": 1, "long": 2}

    df.dropna(axis=0, inplace=True)
    return _to_dict(df=df, key="Token ID", value="Group", mapper=mapper, to_index=to_index)


# Grammatical aspects for verb #########################################################################################


def _to_tense(df_dir: Path, to_index: bool, params: dict):
    """
    Tense, present vs. past
    :param df_dir: directory with .csv files
    :param to_index: if true, convert to indices rather than names
    :param params:
        complex: if true, allow complex verbs
    :return:
        dictionary
    """

    nv_df = pd.read_csv(df_dir / "NV.csv")
    v_df = pd.read_csv(df_dir / "Verbs-Grammatical.csv")

    if not params["complex"]:
        v_df = v_df[v_df["Complex"] is False]

    df = pd.merge(nv_df, v_df, how="left", on="Token ID")
    df.dropna(axis=0, inplace=True)
    return _to_dict(df=df, key="Token ID", value="Tense", mapper={"present": 0, "past": 1}, to_index=to_index)


def _to_person(df_dir: Path, to_index: bool, params: dict):
    """
    Pers, 1 vs. 2 vs. 3
    :param df_dir: directory with .csv files
    :param to_index: if true, convert to indices rather than names
    :param params:
        complex: if true, allow complex verbs
    :return:
        dictionary
    """

    nv_df = pd.read_csv(df_dir / "NV.csv")
    v_df = pd.read_csv(df_dir / "Verbs-Grammatical.csv")

    if not params["complex"]:
        v_df = v_df[v_df["Complex"] is False]

    df = pd.merge(nv_df, v_df, how="left", on="Token ID")
    df.dropna(axis=0, inplace=True)
    return _to_dict(df=df, key="Token ID", value="Person", mapper={1: 0, 2: 1, 3: 2}, to_index=to_index)


def _to_v_number(df_dir: Path, to_index: bool, params: dict):
    """
    Number singular vs. plural (verbs)
    :param df_dir: directory with .csv files
    :param to_index: if true, convert to indices rather than names
    :param params:
        complex: if true, allow complex verbs
    :return:
        dictionary
    """

    nv_df = pd.read_csv(df_dir / "NV.csv")
    v_df = pd.read_csv(df_dir / "Verbs-Grammatical.csv")

    if not params["complex"]:
        v_df = v_df[v_df["Complex"] is False]

    df = pd.merge(nv_df, v_df, how="left", on="Token ID")
    df.dropna(axis=0, inplace=True)
    return _to_dict(df=df, key="Token ID", value="Number", mapper={"sg.": 0, "pl.": 1}, to_index=to_index)


def _to_voice(df_dir: Path, to_index: bool):
    """
    Voice, active vs. passive
    :param df_dir: directory with .csv files
    :param to_index: if true, convert to indices rather than names
    :return:
        dictionary
    """

    nv_df = pd.read_csv(df_dir / "NV.csv")
    v_df = pd.read_csv(df_dir / "Verbs-Grammatical.csv")

    df = pd.merge(nv_df, v_df, how="left", on="Token ID")
    df.dropna(axis=0, inplace=True)
    return _to_dict(df=df, key="Token ID", value="Voice", mapper={"active": 0, "passive": 1}, to_index=to_index)


########################################################################################################################


def _to_gender(df_dir: Path, to_index: bool, params: dict):
    """
    Gender, m. vs. f. vs. n.
    :param df_dir: directory with .csv files
    :param to_index: if true, convert to indices rather than names
    :param params:
    :return:
        diminutives: if true, include diminutives
        combine: if true, combine `f.` and `m.` into a single class `c.
    """

    nv_df = pd.read_csv(df_dir / "NV.csv")
    n_df = pd.read_csv(df_dir / "Nouns-Grammatical.csv")

    if not params["diminutives"]:
        n_df = n_df[n_df["Diminutive"] is False]

    n_df = n_df[n_df["Gender"].isin(["m.", "f.", "n."])]

    if params["combine"]:
        n_df["Gender"] = n_df.apply(lambda x: "c." if "m." or "f." else "n.")
        mapper = {"c.": 0, "n.": 1}
    else:
        mapper = {"m.": 0, "f.": 1, "n.": 2}

    df = pd.merge(nv_df, n_df, how="right", on="Token ID")
    df.dropna(axis=0, inplace=True)
    return _to_dict(df=df, key="Token ID", value="Gender", mapper=mapper, to_index=to_index)


def _to_n_number(df_dir: Path, to_index: bool):
    """
    Number, singular vs. plural (noun)
    :param df_dir: directory with .csv files
    :param to_index: if true, convert to indices rather than names
    :return:
        dictionary
    """

    nv_df = pd.read_csv(df_dir / "NV.csv")
    n_df = pd.read_csv(df_dir / "Nouns-Grammatical.csv")

    n_df = n_df[n_df["Number"].isin(["sg.", "pl."])]

    df = pd.merge(nv_df, n_df, how="right", on="Token ID")
    return _to_dict(df=df, key="Token ID", value="Number", mapper={"sg.": 0, "pl.": 1}, to_index=to_index)
