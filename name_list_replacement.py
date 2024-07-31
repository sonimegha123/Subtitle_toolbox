import logging
import pysrt

from Levenshtein import distance
from flair.models import SequenceTagger
from flair.data import Sentence

def fix_misspelled_name(recognized_name, 
                        confirmed_names, 
                        skip_names, 
                        char_edit_dist_threshold = 2):
    """
    This function aims to repair a recognized name by checking if a similar name exists in the confirmed names list.
    It assumes the passed 'recognized_name' is actually a name as identified by NER. 

    Args:
      recognized_name (str): The name that that the model thinks was said
      confirmed_names (list(str)): A list of confirmed names to check against
      skip_names (list(str)): A list of names to be ignored if found
      char_edit_dist_threshold (int): Threshold for the maximum distance between the recognized name and the confirmed 
   
    returns:
      The best-matching name from the list of confirmed names. If no good match is found, it returns the original recognized name.

    """
  
    # Ignore if name is in the banned list 
    if recognized_name in skip_names:
      return recognized_name

    # Don't change anything if the passed name is in the banned names list 
    potential_names = {}
    for conf_name in confirmed_names:

      # If the difference between the passed name and the confirmed name is small enough
      # store the confirmed name as a potential name
      char_edit_distance = distance(recognized_name, conf_name)

      # If the confirmed name is very short <= 3 letters, use 1 as distance
      if len(conf_name) <= 3 and char_edit_distance <= 1:
          potential_names[conf_name] = char_edit_distance

      # Otherwise use normal distance
      elif len(conf_name) > 3 and char_edit_distance <= char_edit_dist_threshold:
          potential_names[conf_name] = char_edit_distance
      
    # If the potential name list is empty, no replacement is found
    if potential_names == {}:
      return recognized_name
      
    # Otherwise return the name with the lowest edit distance
    sorted_potenial_names = dict(sorted(potential_names.items(), key = lambda x: x[1]))
    best_name = list(sorted_potenial_names.keys())[0]
    logging.info(f"Best replacements for 'recognized_name' sorted_potenial_names")
    return best_name

def recognize_names(text, tagger):
  """
  Recognizes names in the input text using the specified tagger and returns them as a list of dictionaries.

  Args:
    text (string): The input text to recognize names from.
    tagger (Flair[tagger_object]): Specifies which tagger to use.

  Returns:
    A list of dictionaries representing individual recognized names, dictionary has keys 'text', 'tag', 'text_start_pos', and 'text_end_pos'.

  Example:
    >>> recognize_names("John Smith and Mary Brown went to the market together.")
    [{'text': 'John Smith', 'tag': 'PER', 'text_start_pos': 0, 'text_end_pos': 10}, 
    {'text': 'Mary Brown', 'tag': 'PER', 'text_start_pos': 15, 'text_end_pos': 24}]
  """
  # Tokenize the text
  parsed_text = Sentence(text)

  # Predict labels for the tokens 
  tagger.predict(parsed_text)

  # Recognize names and add them to dictionary
  recognized_names = []
  for entity in parsed_text.get_spans('ner'):
    if entity.tag == 'PER':
      recognized_names.append({"text" : entity.text, 
                              "tag" : entity.tag, 
                              "text_start_pos" : entity.start_position,
                              "text_end_pos" : entity.end_position})
  return recognized_names

def fix_misspelled_names_of_caption(caption_text, 
                                    confirmed_names, 
                                    skip_names,
                                    tagger):
  """
  Fixes misspelled person names in a given caption text using NER and a list of confirmed and 'to-skip' names.
  The skipped names will not be changed. 

  Args:
    caption_text (str): The caption text to fix.
    confirmed_names (list(str)): A list of confirmed names to check against
    skip_names (list(str)): A list of names to be ignored if found
    tagger (Flair[tagger_object]): The tagger object to use for NER.

  Returns:
    str: The fixed caption text.

  Example:
    >>> fix_misspelled_names_of_caption("Happy birthday to my friend Johnt!")
    "Happy birthday to my friend John!"
  """
  # Recognized the person names using NER, with tagger Flair object
  recognized_names = recognize_names(text = caption_text, tagger = tagger)
  # print(recognized_names)

  # If no names have been recognized there is nothing to fix 
  if len(recognized_names) == 0:
    return caption_text
  
  # For every recognized name, check if it's a misspelling, and if so, overwrite the text
  fixed_caption = caption_text

  # The position of the recognized names will shift after a name is changed in length
  # so we keep track of how much this will shift
  char_shift = 0
  for rec_name in recognized_names:
    fixed_name = fix_misspelled_name(recognized_name = rec_name['text'], 
                        confirmed_names = confirmed_names, 
                        skip_names = skip_names, 
                        char_edit_dist_threshold = 2)
    
    fixed_caption = fixed_caption[:rec_name['text_start_pos'] + char_shift] + fixed_name + fixed_caption[rec_name['text_end_pos'] + char_shift:]

    # Update shift
    char_shift += len(fixed_name) - len(rec_name['text'])

  return fixed_caption

def fix_misspelled_names_in_srt(srt_path, new_srt_path, tagger):
  """
  Fixes misspelled person names in a .SRT file and saves a new .SRT file with the fixed names.

  Args:
      srt_path (str): The path to the input SRT file.
      new_srt_path (str): The path to the output SRT file.
      tagger (Flair[tagger_object]): The tagger object to use for NER. Defaults to the global tagger object.

  Returns:
      None.

  Example:
      >>> fix_misspelled_names_in_srt("input.srt", "output.srt")
  """
  srt = pysrt.open(srt_path)

  # Count changes 
  change_count = 0
  nr_captions = len(srt)
  for i, sub in enumerate(srt):
    caption = sub.text
    old_caption = caption
    caption = fix_misspelled_names_of_caption(caption, tagger = tagger)
    if old_caption != caption:
      change_count += 1

      # Log the change to be transparent
      old_caption_logtext = old_caption.replace('\n', ' ')
      caption_logtext = caption.replace('\n', ' ')
      logging_text = f"Caption {i}/{nr_captions} from {sub.start} to {sub.end}:\nBefore: '{old_caption_logtext}'\nAfter:  '{caption_logtext}'\n"
      logging.debug(logging_text)

      # Replace caption
      sub.text = caption

  print(f"Changed {change_count} out of {len(srt)} captions")
  logging.info(f"Changed {change_count} out of {len(srt)} captions")

  # Overwrite
  srt.save(new_srt_path)

def get_default_tagger():
  """ Get the default NER tagger for dutch with the best performance.

  Returns:
      A flair SequenceTagger object.
  """
  return SequenceTagger.load('flair/ner-dutch-large')