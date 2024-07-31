import pysrt
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def get_default_substyle() -> None:
    """ Returns the default substyle dictionary.

    Returns:
        dict: A dictionary containing the default substyle.
    """
    return {"max_chars_per_caption" : 74,
            "max_chars_per_line" : 37,
            "max_seconds_per_caption" : 8.0,
            "min_seconds_per_caption" : 1.5,
            "highest_written_form_number" : 10,
            "add_ellipses" : True,
            "translate" : True,
            "remove_audio_descriptions" : True, }

def df_to_subtitle_fig(df: pd.DataFrame,
                      name: str = "Subtitles",
                      color: str = "Blue",
                      offset: int = 1,
                      flat_line: bool = True) -> go.Figure:
    """ Plots subtitles in a Plotly Figure. This base figure can be extended upon.

    Args:
        df (pd.DataFrame): The subtitle dataframe to base the figure on.
        name (str): Name for the subtitle line in the graph. Defaults to "Subtitles".
        color (str): Color of the subtitle line. Defaults to "Blue".
        offset (int): Height offset for the verticalty of the subtitle line. Defaults to 1.
        flat_line (bool): If the subtitle line should be flat. Defaults to True.

    Returns:
        A plotly.graph_objects.Figure() figure object.
    """
    fig = go.Figure()

    # X-axis = [st_0, et_0, None, st_1, et_1, None, st_n, et_n, None]
    plotly_sub_x = []
    for st, et in zip(df["start_time"], df["end_time"]):
        plotly_sub_x.extend([st, et, None])

    # Y-axis = [0, 0, None, 1, 1, None, 2, 2, None]
    i = offset
    plotly_sub_y = []
    for v in np.arange(len(df)):
        plotly_sub_y.extend([i, i, None])
        if not flat_line and v%2==0:
            i = offset + 1
        else:
            i = offset

    # Text per subtitle
    plotly_sub_text = []
    plotly_sub_nr_chars = []
    plotly_sub_duration = []
    plotly_color = []
    for text, st, et, hrst in zip(df["text"], df["start_time"], df["end_time"], df['human_start_time']):
        duration = et - st
        nr_chars = len(text)
        text_entry = {"text": text, "duration": np.round(duration, 3), "nr_chars": nr_chars, "hrst": hrst}
        plotly_sub_text.extend([text_entry, text_entry, None])
        plotly_sub_nr_chars.extend([nr_chars, nr_chars, None])
        plotly_sub_duration.extend([plotly_sub_duration, plotly_sub_duration, None])
        plotly_color.extend(['darkblue', 'darkred', 'antiquewhite'])

    fig.add_trace(
        go.Scatter(
            x=plotly_sub_x,
            y=plotly_sub_y,
            mode="lines",
            name=name,  # Style name/legend entry with html tags
            connectgaps=False,  # override default to connect the gaps
            text=plotly_sub_text,
            marker=dict(color=color),
        )
    )

    return fig

def plot_subs_with_characteristics(df: pd.DataFrame, title: str = "Subtitles", substyle: dict = None) -> None:
    """
    Plots subtitles in a Plotly Figure with characteristics based on the substyle

    Args:
        df (pandas.DataFrame): A Pandas dataframe containing subtitle details.
        title (str, optional): The title to display on the plot. Defaults to "Subtitles".
        substyle (dict, optional): A dictionary containing the substyle. Defaults to None (uses the default substyle).

    Returns:
        None

    """
    
    # Get default style
    if substyle is None:
        substyle = get_default_substyle()

    # Data prep
    df = df.rename(columns = {"sentence" : "text"})  # if from caption dict, this triggers
    df["duration"] = df["end_time"] - df["start_time"]
    df["midpoint"] = ((df["end_time"] - df["start_time"]) / 2) + df["start_time"]
    df["nr_chars"] = df["text"].apply(lambda x: len(x))

    df["too_long"] = df["duration"] > substyle["max_seconds_per_caption"]
    df["too_short"] = (df["duration"] < substyle["min_seconds_per_caption"]) & (df["duration"] >= 1.0)
    df["way_too_short"] = df["duration"] < 1.0
    df["too_many_chars"] = df["nr_chars"] > substyle["max_chars_per_caption"]

    fig = df_to_subtitle_fig(df, name = title, flat_line = True)

    # Scatterplots for CPS
    if "char_per_sec" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['start_time'],
                y=df['char_per_sec'],
                mode="markers",
                name="CPS",  # Style name/legend entry with html tags
                connectgaps=False,  # override default to connect the gaps
                marker=dict(size=8, line=dict(width=1, color=df['char_per_sec']), symbol="circle"),
                text = df['human_start_time'] + ' -> ' + df['human_end_time']
            )
        )

    # Scatterplots for mistakes
    if "too_long" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[df["too_long"]]["midpoint"],
                y=[0] * len(df),
                mode="markers",
                name="Too Long",
                connectgaps=False,
                marker=dict(size=8, line=dict(width=1, color="Red"), symbol="x"),
                line=dict(color=  "Red"),
            )
        )

    if "too_short" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[df["too_short"]]["midpoint"],
                y=[0] * len(df),
                mode="markers",
                name=f"Too Short (< {substyle['min_seconds_per_caption']}s)",
                connectgaps=False,
                marker=dict(size=8, line=dict(width=1, color="Red"), symbol="x"),
                line=dict(color="Orange"),
            )
        )

    if "way_too_short" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[df["way_too_short"]]["midpoint"],
                y=[0] * len(df),
                mode="markers",
                name="Way Too Short (< 1.0s)",
                connectgaps=False,
                marker=dict(size=8, line=dict(width=1, color="Red"), symbol="x"),
                line=dict(color="Yellow"),
            )
        )

    if "too_many_chars" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[df["too_many_chars"]]["midpoint"],
                y=[0] * len(df),
                mode="markers",
                name="Too Many Characters",
                connectgaps=False,
                marker=dict(size=8, line=dict(width=1, color="Red"), symbol="x"),
                line=dict(color="Blue"),
            )
        )

    fig.show()
