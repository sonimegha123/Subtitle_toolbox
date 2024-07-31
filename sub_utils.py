import pandas as pd
import pysrt
from typing import Union
from nltk.tokenize import RegexpTokenizer
import jiwer
import numpy as np
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz


def srt_equals(srt_a: pysrt.SubRipFile, srt_b: pysrt.SubRipFile):
  """ Compares if two pysrt objects are equal to each other in terms of content.

  Args:
      srt_a: First pysrt object to compare.
      srt_b: Second pysrt object to compare.

  Returns:
      bool: True if the two objects are equal, False otherwise.
  """
  # Quick check
  if len(srt_a) != len(srt_b):
    return False
  
  # Slower check
  for a, b in zip(srt_a, srt_b):
    if a.text != b.text:
      return False
    if a.start != b.start:
      return False
    if a.end != b.end:
      return False
  
  return True

def df_to_srt(df: pd.DataFrame) -> pysrt.SubRipFile:
  """ Converts a subtitle dataframe to a pysrt object.
  
  Args:
    df (DataFrame): A subtitle dataframe with columns 'text', 'start_time' and 'end_time'.
  
  Returns:
    pysrt.SubRipFile: A pysrt subtitle object.
  """
  items = []
  for text, st, et in zip(df['text'], df['start_time'], df['end_time']):
    item = pysrt.SubRipItem()
    item.text = text
    item.start.seconds = st
    item.end.seconds = et
    items.append(item)
  return pysrt.SubRipFile(items = items)

def srt_to_df(srt_obj: pysrt.SubRipFile) -> pd.DataFrame:
  """ Converts a pysrt object to a subtitle dataframe.

  Args:
    srt_obj (pysrt.SubRipFile): A pysrt subtitle object.

  Returns:
    DataFrame: A subtitle dataframe with columns 'text', 'start_time', 'end_time', 'human_start_time' and 'human_end_time'.
  """
  return pd.DataFrame([{"text": item.text, 
    "start_time": item.start.ordinal/1000, 
    "end_time": item.end.ordinal/1000,
    "human_start_time" : str(item.start),
    "human_end_time" : str(item.end),
    } for item in srt_obj])

def srt_to_clean_text(srt_obj: pysrt.SubRipFile) -> str:
  """ Cleans the srt text by replacing linebreaks and line continuations.

  Args:
    srt_obj (pysrt.SubRipFile): The SRT object to be displayed, which can be generated from an SRT file using the `pysrt.open()` method.

  Returns:
    (str): The cleaned text

  """ 
  text = srt_obj.text
  text = text.replace("\n", " ")
  text = text.replace("...", "")
  return text


def print_srt(pysrt_obj: pysrt.SubRipFile) -> None:
  """
  Prints the start and end times along with the text of each subtitle in an SRT object.

  Args:
    pysrt_obj (pysrt.SubRipFile): The SRT object to be displayed, which can be generated from an SRT file using the `pysrt.open()` method.

  """
  for caption in pysrt_obj:

    # Linebreaks look odd during print so we replace them
    text = caption.text.replace('\n', ' <LB> ')
    print(f"{caption.start} -> {caption.end} {text}")


def detect_and_fix_hour_shift_srt(srt: Union[str, pysrt.SubRipFile]) -> None:
  """
  Detects and fixes a time shift of 1 hour in an SRT file with respect to the first subtitle start time.

  Args:
    srt (str | pysrt.SubRipFile): The srt object or file path for the SRT file to fix.

  """
  if isinstance(srt, str):
    srt_object = pysrt.open(srt)
  else:
    srt_object = srt

  first_sub_start_time_hour = srt_object[0].start.hours

  if first_sub_start_time_hour >= 1:
    srt_object.shift(hours = -1)
  return srt_object


def add_subtitle_features(df: pd.DataFrame) -> pd.DataFrame:
  """ Add subtitle features like number of characters, duration and characters per second to a subtitle dataframe.

  Args:
    DataFrame (df): A subtitle dataframe with columns 'text', 'start_time' and 'end_time'.

  Returns:
    DataFrame: The subtitle dataframe with added features 'nr_chars', 'duration' and 'char_per_sec'.
  """
  df['nr_chars'] = df['text'].apply(lambda x: len(x))
  df['duration'] = df['end_time'] - df['start_time']
  df['char_per_sec'] = df['nr_chars']/df['duration']
  return df


def normalize_text(pysrt_obj: pysrt.SubRipFile) -> None:
    """
    Normalizes the text of each subtitle in an SRT object and prints the start and end times along with the normalized text.

    Args:
        pysrt_obj (pysrt.SubRipFile): The SRT object to be normalized, which can be generated from an SRT file using the `pysrt.open()` method.

    """
    for caption in pysrt_obj:
        tokenizer = RegexpTokenizer(r'\w+')
        tokenized_sent = tokenizer.tokenize(caption.text)
        words = []

        for word in tokenized_sent:
            words.append(word.lower())
        normalized_text = ' '.join(words)
        print(f"{caption.start} -> {caption.end} {normalized_text}")


def calculate_similarity_metrics(srt_a: pysrt.SubRipFile, srt_b: pysrt.SubRipFile) -> dict:
    """
    Calculate various text similarity metrics using the jiwer library.

    Args:
        srt_a: The reference (ground truth) subtitles.
        srt_b: The hypothesis (predicted) subtitles.

    Returns:
        dict: A dictionary containing the calculated similarity metrics.
    """
    similarity_metrics = {}

    ref_text = srt_to_clean_text(srt_a)
    hypo_text = srt_to_clean_text(srt_b)

    wer = jiwer.wer(ref_text, hypo_text)
    mer = jiwer.mer(ref_text, hypo_text)
    cer = jiwer.cer(ref_text, hypo_text)
    wil = jiwer.wil(ref_text, hypo_text)
    wip = jiwer.wip(ref_text, hypo_text)
    similarity_metrics["Word Error Rate"] = wer
    similarity_metrics["Merge Error Rate"] = mer
    similarity_metrics["Character Error Rate"] = cer
    similarity_metrics["Word Information Lost"] = wil
    similarity_metrics["Word Information Preserved"] = wip

    return similarity_metrics


def visualize_text_alignment(srt_a: pysrt.SubRipFile, srt_b: pysrt.SubRipFile) -> None:
  """ 
  shows alignment between each pair of pysrt objects along with the count of substitutions, deletions, insertions and hits
  
  Args:
      srt_a: The reference (ground truth) subtitles.
      srt_b: The hypothesis (predicted) subtitles.
  """
  if len(srt_a) != len(srt_b):
        raise ValueError("Input subtitle files must have the same number of entries.")
    
  for a, b in zip(srt_a, srt_b):
    out = jiwer.process_words(a.text, b.text)
    print(jiwer.visualize_alignment(out))


def time_to_seconds(time):
    """
    Converts a pysrt time to seconds (floating-point).
    """
    return time.hours * 3600 + time.minutes * 60 + time.seconds + time.milliseconds / 1000.0


def calculate_timing_alignment_errors(srt_a: pysrt.SubRipFile, srt_b: pysrt.SubRipFile) -> list[tuple[float, float]]:
    """
    Calculates timing alignment errors between two pysrt subtitle objects.

    Args:
        srt_a: The reference (ground truth) subtitles.
        srt_b: The hypothesis (predicted) subtitles.

    Returns:
        List of tuples: Each tuple contains the start and end time errors (in seconds) for corresponding subtitles.
                         The list represents the timing alignment errors between srt_a and srt_b.
    """

    alignment_errors = []
    for a, b in zip(srt_a, srt_b):
        start_time_ref = time_to_seconds(a.start)
        end_time_ref = time_to_seconds(a.end)
        start_time_hyp = time_to_seconds(b.start)
        end_time_hyp = time_to_seconds(b.end)

        start_error = start_time_ref - start_time_hyp
        end_error = end_time_ref - end_time_hyp

        alignment_errors.append((start_error, end_error))

    return alignment_errors


def get_alignments(srt_a: pysrt.SubRipFile, srt_b: pysrt.SubRipFile) -> list:
  """ 
  Gives alignment information for each pair of pysrt objects along with the indices of substitutions, deletions and insertions.
  
  Args:
      srt_a: The reference (ground truth) subtitles.
      srt_b: The hypothesis (predicted) subtitles.
  """
  if len(srt_a) != len(srt_b):
        raise ValueError("Input subtitle files must have the same number of entries.")

  all_alignments = []

  for a, b in zip(srt_a, srt_b):
    out = jiwer.process_words(a.text, b.text)
    alignments = out.alignments
    all_alignments.append(alignments)

  return all_alignments

def display_misplaced_words(srt_a: pysrt.SubRipFile, srt_b: pysrt.SubRipFile) -> None:
    alignments = get_alignments(srt_a, srt_b)
    reference = srt_to_clean_text(srt_a)
    hypothesis = srt_to_clean_text(srt_b)

    ref_words = word_tokenize(reference)
    hyp_words = word_tokenize(hypothesis)

    for chunk_list in alignments:
      for alignment in chunk_list:
        if alignment.type == 'insert':
            inserted_word = hyp_words[alignment.hyp_start_idx]
            print(f"Type: {alignment.type}")
            print(f"Inserted Word: {inserted_word}")
            print(f"Hypothesis Word Index: {alignment.hyp_start_idx}")
            print()
        elif alignment.type == 'substitute':
            substituted_word_ref = ref_words[alignment.ref_start_idx]
            substituted_word_hyp = hyp_words[alignment.hyp_start_idx]
            print(f"Type: {alignment.type}")
            print(f"Substituted Word (Reference): {substituted_word_ref}")
            print(f"Substituted Word (Hypothesis): {substituted_word_hyp}")
            print(f"Reference Word Index: {alignment.ref_start_idx}")
            print(f"Hypothesis Word Index: {alignment.hyp_start_idx}")
            print()
        elif alignment.type == 'delete':
            deleted_word = ref_words[alignment.ref_start_idx]
            print(f"Type: {alignment.type}")
            print(f"Deleted Word: {deleted_word}")
            print(f"Reference Word Index: {alignment.ref_start_idx}")
            print()


def text_similarity(text_a: str, text_b: str) -> int:
    """ Calculate the text similarity using fuzzy ratio. It works by measuring the 
        similarity between two strings based on the number of insertions, deletions, 
        and substitutions required to transform one string into the other.

    Args:
        text_a (str): First text string to compare.
        text_b (str): Second text string to compare.

    Returns:
        int: Fuzzy ratio representing the similarity between the two strings.
    """
    
    return fuzz.ratio(text_a, text_b) 
    
  
