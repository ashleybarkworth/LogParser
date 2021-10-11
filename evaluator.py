import pandas as pd
import scipy.special


def evaluate(groundtruth, parsedresult):
    """
    Evaluates and prints log parsing accuracy
    :param groundtruth: file path of groundtruth structured csv file
    :param parsedresult: file path of parsed structured csv file
    :return: accuracy : float
    """
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)
    non_empty_log_ids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index
    df_groundtruth = df_groundtruth.loc[non_empty_log_ids]
    df_parsedlog = df_parsedlog.loc[non_empty_log_ids]
    accuracy = get_accuracy(df_groundtruth['EventId'], df_parsedlog['EventId'])
    return accuracy


def get_accuracy(series_groundtruth, series_parsedlog):
    """

    :param series_groundtruth: A sequence of groundtruth event Ids
    :param series_parsedlog: A sequence of parsed event Ids
    :return: accuracy: float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        log_ids = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_log_id_valuecounts = series_groundtruth[log_ids].value_counts()
        error_event_ids = (parsed_eventId, series_groundtruth_log_id_valuecounts.index.tolist())
        error = True
        if series_groundtruth_log_id_valuecounts.size == 1:
            groundtruth_event_id = series_groundtruth_log_id_valuecounts.index[0]
            if log_ids.size == series_groundtruth[series_groundtruth == groundtruth_event_id].size:
                accurate_events += log_ids.size
                error = False
        for count in series_groundtruth_log_id_valuecounts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    accuracy = float(accurate_events) / series_groundtruth.size
    return accuracy
