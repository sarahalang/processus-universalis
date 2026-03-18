import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

import re
import unicodedata, ast

from pathlib import Path
import os
import time

from utils import clean_filename

DATA_DIR = Path(__file__).parent / "ProcessusUniversalis_relevant-files-for-2025"
RESULT_DIR = Path(__file__).parent / "results"
PROCESSED_DATA_DIR = Path(__file__).parent / "processed_data"

TXT_FILES_DIR = DATA_DIR / "txt-files-lowercase_processus"

class PhyloXMLParser:
    """Parser for PhyloXML files to extract phylogenetic tree data."""
    XML_FILE = DATA_DIR / Path("sammlung_aller_texte.xml") #.with_suffix(".xml")
    if not XML_FILE.exists():
        raise FileNotFoundError(f"XML file not found: {XML_FILE}")
    def __init__(self, xmlpath: str|Path = XML_FILE, timestamp: bool = True):
        self.timestamp = time.strftime("%Y%m%d_%H%M", time.localtime()) # if timestamp else ""

        self.xmlpath = Path(xmlpath)
        self.tree = ET.parse(xmlpath)
        self.root = self.tree.getroot()

        splitstree_dir = Path("splitstree") / self.timestamp if timestamp else Path("splitstree")
        self.processpath = PROCESSED_DATA_DIR / splitstree_dir
        self.resultpath = RESULT_DIR / splitstree_dir
        self.processpath.mkdir(parents=True, exist_ok=True)
        self.resultpath.mkdir(parents=True, exist_ok=True)


    def get_clades(self):
        """Extract clade information from the PhyloXML file."""
        clades = []
        for clade in self.root.findall(".//clade"):
            clade_info = {
                "name": clade.findtext("name"),
                "branch_length": clade.findtext("branch_length"),
                "confidence": clade.findtext("confidence"),
            }
            clades.append(clade_info)
        return pd.DataFrame(clades)

    def get_sequences(self):
        """Extract sequence information from the PhyloXML file."""
        sequences = []
        for seq in self.root.findall(".//sequence"):
            seq_info = {
                "id": seq.get("id"),
                "type": seq.get("type"),
                "value": seq.text,
            }
            sequences.append(seq_info)
        return pd.DataFrame(sequences)

    # ---------------------------------------------------------
    # HELPER FUNCTION: THE TEXT STITCHER
    # ---------------------------------------------------------
    def _get_full_text(self, element):
        """
        Reconstructs the full text of a mixed-content element,
        grabbing text inside tags AND text between tags (tails).
        """
        text_content = []
        
        # 1. Text inside the current element (before the first child)
        if element.text:
            text_content.append(element.text.strip())
            
        # 2. Loop through all children
        for child in element:
            # Recursively get text inside the child (e.g., <hi>underlined</hi>)
            text_content.append(self._get_full_text(child))
            
            # 3. Text AFTER the child (the "tail"), before the next tag starts
            if child.tail:
                text_content.append(child.tail.strip())
                
        return " ".join(filter(None, text_content))
        
    # ---------------------------------------------------------
    # MAIN EXTRACTION SCRIPT
    # ---------------------------------------------------------
    def get_df(self, out: None|str|Path = None):
        data = []
        keywords_set = set([kw.get('type') for kw in self.root.find('keywords').findall('keyword')])
        keys_set = set() # To track found keys; initialized as empty set

        # Iterate over every recipe (div)
        warning_corpuses = []
        for div in self.root.findall('div'):
            row = {}
        
            # --- 1. METADATA FROM THE DIV ITSELF ---
            row['Corpus_ID'] = div.get('type')  # e.g., g1a1
            row['Full_Title'] = div.get('n')    # e.g., A1 Höchster Schatz...
        
            # --- 2. EXTRACT THE KEYS (The Structured Data) ---
            # We find all keys, regardless of how deep they are buried
            for key in div.iter('keys'):
                col_name = key.get('type')
                val_name = key.get('n')
            
                # Clean the semicolon and spaces
                if val_name:
                    val_name = unicodedata.normalize('NFC', val_name).strip() # string instead of list, using regex later
                    # val_name = [v.strip() for v in val_name.split(';') if v.strip()]
                
                # Handle the "FEHLT" (Missing) logic (Handle later with regex)
                if "FEHLT" in val_name:
                    val_name = "FEHLT;"
                
                # Add to row (if column exists, it overwrites - acceptable here)
                if col_name:
                    row[col_name] = val_name
                    keys_set.add(col_name)
                else:
                    warning_corpuses.append(row['Corpus_ID'])
                    # print(f"Warning: Found a <keys> without a 'type' attribute in Corpus_ID {row['Corpus_ID']}")
        
            # # --- 3. EXTRACT THE FULL READABLE TEXT ---
            # No need since we already have original txt files
            # # We use our helper function to stitch the story back together
            # row['Full_Text'] = self._get_full_text(div)
            data.append(row)

        if warning_corpuses:
                print(f"Warning: Found <keys> without 'type' attribute in the following Corpus_IDs: {', '.join(warning_corpuses)}")


        # Check if found keys match keywords set
        if keys_set != keywords_set:
            print("Warning: Mismatch between found keys and master keywords!")
            print("Found keys:", keys_set)
            print("Master keywords:", keywords_set)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Reorder columns: ID and Text first, then the extracted keys
        cols = ['Corpus_ID', 'Full_Title'] + [c for c in df.columns if c not in ['Corpus_ID', 'Full_Title', 'Full_Text']]
        df = df[cols]

        if out:
            outpath = Path(out)
            if not outpath.parent.exists():
                print(f"{outpath} not exists. Setting outpath to './characters.csv'")
                outpath = Path('./characters.csv')
            df.to_csv(outpath, sep=';')
            print(f"DataFrame saved to {outpath}")

        self.df = df
        # return df

    @staticmethod
    def match_filenames(old_filenames, new_filenames, xml_df):
        columns = ['Old_Filename', 'New_Filename', 'Full_Title']
        matched_data = []
        # deep copy new_filenames, xml_titles to avoid modifying original lists
        new_filenames_copy = new_filenames.copy()
        xml_titles_copy = xml_df["Full_Title"].tolist().copy()
        xml_ids_copy = xml_df["Corpus_ID"].tolist().copy()

        # get re for filenames
        re_oldfilenames = [clean_filename(fn) for fn in old_filenames]
        re_newfilenames = [clean_filename(fn) for fn in new_filenames_copy]
        # re_xml_titles = [clean_filename(title) for title in xml_titles_copy]

        for old_fn, (old_id, re_old_fn) in zip(old_filenames, re_oldfilenames):
            # print(f"Processing old filename: {old_fn} with cleaned id: {old_id} and cleaned name: {re_old_fn}")
            match_new_fn = None
            for new_fn, (new_id, re_new_fn) in zip(new_filenames_copy, re_newfilenames):
                if re_old_fn == re_new_fn:
                    # find index and pop from new_filenames_copy and re_newfilenames
                    index = new_filenames_copy.index(new_fn)
                    # match_new_fn = new_fn
                    match_new_fn = new_filenames_copy.pop(index)
                    re_newfilenames.pop(index)
                    break
            if not match_new_fn:
                print(f"Warning: No match found between 'new filename' and 'old filename': `{old_fn}`")
            
            match_title = None
            for title, xml_id in zip(xml_titles_copy, xml_ids_copy):
                if old_id == xml_id:
                    match_title = title
                    # find index and pop from xml_titles_copy and xml_ids_copy
                    index = xml_titles_copy.index(title)
                    xml_titles_copy.pop(index)
                    xml_ids_copy.pop(index)
                    break
            if not match_title:
                print(f"Warning: No match found between 'title' and 'old filename': `{old_fn}`")
            
            matched_data.append([old_fn, match_new_fn, match_title])
        
        return pd.DataFrame(matched_data, columns=columns)

    def get_file_mapping(self,
            # work_dir: str|Path = DATA_DIR, # "/Users/tchang/sites/processus_universalis/ProcessusUniversalis_relevant-files-for-2025"
    ):
        texts_dir = TXT_FILES_DIR
        xml_filename_file = DATA_DIR / "original_filenames_txtHasNew_XMLusesOld.txt"
        if not xml_filename_file.exists():
            print(f"Warning: {xml_filename_file} does not exist. Cannot proceed with file mapping.")
            self.file_mapping = None

        with open (xml_filename_file, 'r') as f:
            lines = f.readlines()
            xml_filenames = [line.strip() for line in lines]
        
        new_filenames = os.listdir(texts_dir)

        self.file_mapping = self.match_filenames(
            old_filenames = xml_filenames,
            new_filenames = new_filenames,
            xml_df = self.df,
        )
        # return self.file_mapping


    def dict_old2new_fileids(self) -> dict:
        """
        Read file mapping df and return dict {old_fileid: new_fileid}.
        - robust to NaNs
        - applies regex replacement elementwise using pandas string methods
        """
        # mapping = Path(mapping)
        # df_map = pd.read_csv(mapping, sep=';')
        if not hasattr(self, 'file_mapping'):
            self.get_file_mapping()
        df_map = self.file_mapping
        if df_map is None:
            # print("Warning: file_mapping is not available. Returning empty dict.")
            return {}

        # sanity checks
        if "Old_Filename" not in df_map.columns or "New_Filename" not in df_map.columns:
            raise ValueError("filename_mapping.csv must contain columns 'Old_Filename' and 'New_Filename'")

        # helper to extract fileid (clean_filename returns (fileid, filename))
        def _fileid_from(val):
            if pd.isna(val):
                return ""
            return clean_filename(str(val))[0] # or ""

        oldids = df_map["Old_Filename"].map(_fileid_from).astype(str).str.strip()
        newids = df_map["New_Filename"].map(_fileid_from).astype(str).str.strip()
        # remove g<digits> tokens from newids (apply elementwise)
        # newids = newids.str.replace(r"g\d+", "", regex=True).str.strip()

        # optional: warn about duplicate oldids
        if oldids.duplicated().any():
            dup = oldids[oldids.duplicated()].unique().tolist()
            print(f"Warning: duplicate oldids found in mapping: {dup} (last occurrence will be used)")

        return dict(zip(oldids.tolist(), newids.tolist()))
    
    @staticmethod
    def replace_with_digit_pattern(s: str) -> str:
        """ Replace digit patterns in a string with a regex pattern and a human-readable pattern.
        Test cases:
        >>> tests = [
                "3 Stunden schmelzen",
                "6 Stunden schmelzen",
                "3. Überschrift",
                "2.2 Kapitel",
                "5: Aufzählung",
                "im Tiegel 7 Monate schmelzen",
                "1a. 40-44 Tage im Wasserdampf",
                "1a. 40-45 Tage im Wasserdampf",
                "1a. 43 Tage im Wasserdampf",
                "2e: nach 50 Tagen im Aschebad: Blutrot mit Rubinkorn",
            ]
        >>> for test in tests:
        >>>     print(t, "=>", replace_with_digit_pattern(t))
        Output:
        >>> 3 Stunden schmelzen => ('\\d+,?\\d*(-\\d+,?\\d*)? Stunden schmelzen', '[NUM] Stunden schmelzen')
        >>> 6 Stunden schmelzen => ('\\d+,?\\d*(-\\d+,?\\d*)? Stunden schmelzen', '[NUM] Stunden schmelzen')
        >>> 3. Überschrift => ('3. Überschrift', '3. Überschrift')
        >>> 2.2 Kapitel => ('2.2 Kapitel', '2.2 Kapitel')
        >>> 5: Aufzählung => ('5: Aufzählung', '5: Aufzählung')
        >>> im Tiegel 7 Monate schmelzen => ('im Tiegel \\d+,?\\d*(-\\d+,?\\d*)? Monate schmelzen', 'im Tiegel [NUM] Monate schmelzen')
        >>> 1a. 40-44 Tage im Wasserdampf => ('1a. \\d+,?\\d*(-\\d+,?\\d*)? Tage im Wasserdampf', '1a. [NUM] Tage im Wasserdampf')
        >>> 1a. 40-45 Tage im Wasserdampf => ('1a. \\d+,?\\d*(-\\d+,?\\d*)? Tage im Wasserdampf', '1a. [NUM] Tage im Wasserdampf')
        >>> 1a. 43 Tage im Wasserdampf => ('1a. \\d+,?\\d*(-\\d+,?\\d*)? Tage im Wasserdampf', '1a. [NUM] Tage im Wasserdampf')
        >>> 2e: nach 50 Tagen im Aschebad: Blutrot mit Rubinkorn => ('2e: nach \\d+,?\\d*(-\\d+,?\\d*)? Tagen im Aschebad: Blutrot mit Rubinkorn', '2e: nach [NUM] Tagen im Aschebad: Blutrot mit Rubinkorn')
        """
        # raw pattern (no SyntaxWarning)
        pattern = r'(?<!\S)(?![^\s]*[.:])\d+(?:,\d*)?(?:-\d+(?:,\d*)?)?(?![.:])'
        # If you want the replacement to be the literal regex fragment "\d+,?\d*"
        # (i.e. text containing a backslash), use a raw string for the replacement too:
        # repl = r'\d+,?\d*'   # yields the string '\\d+,?\\d*' in Python but prints as '\d+,?\d*'
        repl = r'\d+,?\d*(-\d+,?\d*)?'
        return re.sub(pattern, lambda m: repl, s), re.sub(pattern, r'[NUM]', s)

    # @staticmethod
    def _get_character_dicts(self,
        df: pd.DataFrame,
        title_cols: list[str] = ["Corpus_ID", "Full_Title"]
    ):
        character_cols = [col for col in df.columns if col not in title_cols]
        ## binary characters: those with exactly 2 unique values (including NaN)
        character_two_unique = df.columns[df.nunique(dropna=False) == 2]
        binary_character_dict = {}
        for char in character_two_unique:
            binary_character_dict[char] = df[char].unique() # unique_vals
        ## rest characters: those with more than 2 unique values
        rest_characters = [char for char in character_cols if char not in character_two_unique]
        rest_character_dict = {}
        for char in rest_characters:
            unique_vals = set()
            for raw_str in df[char].unique():
                if pd.isna(raw_str): # skip NaN (Missing description of this character)
                    continue
                if raw_str == 'FEHLT;': # "Missing" is not a keyword, pass
                    continue
                keywords = re.split(r';\s*', str(raw_str))
                for kw in keywords:
                    kw = kw.strip()
                    if kw and kw != 'FEHLT': # check again to avoid 'FEHLT'
                        unique_vals.add(kw)
            rest_character_dict[char] = unique_vals
        
        ## For each string in rest_character_dict, turn all non-head digits into regex patterns that match the whole string. Similar recipes with different numbers will then be captured by the same binary column. Also produce a human-readable version of the string with [NUM] placeholder.
        tmp_dict = {}
        for char, val_set in rest_character_dict.items():
            re_val_set = set() # create a new set to hold unique regex patterns and human-readable patterns
            for val in val_set:
                re_val, human_val = self.replace_with_digit_pattern(val)
                re_val_set.add((re_val, human_val))
            tmp_dict[char] = re_val_set
        rest_character_dict = tmp_dict
        
        return binary_character_dict, rest_character_dict

    def get_binary_matrix_nexus(self,
        # df: pd.DataFrame,
        # binary_character_dict: dict,
        # rest_character_dict: dict,
        error_char: str = "?",
        taxa_col: str = "Corpus_ID",
        nexus_path: str | Path = "characters.nex",
        mapping_path: str | Path = "characters_mapping.csv",
        binary_path: str | Path = "characters_binary.csv",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build binary matrix and write NEXUS file. Also produce a mapping CSV that records
        the human-readable meaning of 1/0 (and error) for every binary output column.

        Returns (binary_df, mapping_df).

        Mapping rules:
        - For binary_character_dict: determine the 'positive' label (first non-"None" if present).
            If "None" is present in the allowed values it will be considered the "absent/NaN" label.
            mapping entry for that column will be:
            column, label_1, label_0, label-error
        - For rest_character_dict -> each produced column is `key__value`. label_1 = "present", label_0 = "absent".
        - label_error is set to `error_char` (usually "?").
        """
        nexus_path = Path(nexus_path)
        mapping_path = Path(mapping_path)
        df = self.df.copy()

        ## Prepare binary_character_dict and rest_character_dict
        binary_character_dict, rest_character_dict = self._get_character_dicts(df)

        # setting up columns from character dicts
        binary_cols = binary_character_dict.keys() # [str(c) for c in binary_character_dict.keys()]
        multi_cols_info = []
        for char, val_set in rest_character_dict.items():
            for pattern, str_readable in val_set:
                # safe_v = str(v).replace(" ", "_").replace(";", "_").replace("/", "_")
                multi_cols_info.append((char, pattern, str_readable))

        # Build mapping rows
        mapping_rows = []

        # prepare mapping for binary columns
        binary_mappings = []
        for char, vals in binary_character_dict.items():
            col = str(char)
            label_0, label_1 = vals[0], vals[1]
            for i in range(2):
                if str(vals[i]).lower() in ["fehlt;", "nein;"]:
                    label_0, label_1 = vals[i], vals[1 - i]
                    break
            
            binary_mappings.append((char, {label_0: 0, label_1: 1}))

            mapping_rows.append({
                "column": col,
                "label_0": label_0,
                "label_1": label_1,
                "label_error": error_char
            })
        # prepare mapping for multi-value columns (one col per (char,value))
        for char, pattern, str_readable in multi_cols_info:
            mapping_rows.append({
                "column": f"{char}__{str_readable}",
                "label_0": "absent",
                "label_1": str_readable,
                "label_error": error_char
            })

        # rows_bits = []
        rows_bits = np.empty((len(df), len(binary_cols) + len(multi_cols_info)), dtype=str)
        for j, (char, vals) in enumerate(binary_mappings):
            rows_bits[:, j] = df[char].map(
                lambda content: str(vals[content]) if content in vals else error_char
            ).astype(str).values
        
        ncol_binary = len(binary_cols)
        for j, (char, pattern, _) in enumerate(multi_cols_info):
            # def map_multi(content):
            #     if pd.isna(content) or content == "":
            #         return error_char
            #     if re.search(pattern, str(content)):
            #         return "1"
            #     else:
            #         return "0"
            rows_bits[:, ncol_binary + j] = df[char].map(
                lambda content: error_char if pd.isna(content) or content == "" else ("1" if re.search(pattern, str(content)) else "0")
            ).astype(str).values

        # all_col_order = binary_cols + [f"{char}__{str_read}" for (char,_,str_read) in multi_cols_info]
        all_col_order = [mapping_row["column"] for mapping_row in mapping_rows]
        idxs = df[taxa_col].astype(str).tolist()
        dict_oldid2newid = self.dict_old2new_fileids()
        if dict_oldid2newid:
            new_idxs = [dict_oldid2newid.get(idx, '') for idx in idxs]
            combined_indices = [f"{idx}-{newid}" for idx, newid in zip(idxs, new_idxs)]
        else:
            combined_indices = idxs
        # binary_df = pd.DataFrame([list(s) for s in rows_bits], columns=all_col_order, index=combined_indices) # index=df[taxa_col].astype(str).tolist()
        binary_df = pd.DataFrame(rows_bits, columns=all_col_order, index=combined_indices) # index=df[taxa_col].astype(str).tolist()

        # write NEXUS
        ntax = len(binary_df)
        nchar = len(all_col_order)
        with nexus_path.open("w", encoding="utf-8") as f:
            f.write("#NEXUS\n")
            f.write("Begin data;\n")
            f.write(f"  Dimensions ntax={ntax} nchar={nchar};\n")
            f.write("  Format datatype=standard symbols=\"01\" gap=- missing=?;\n")
            f.write("Matrix\n")
            # get the max length of taxon labels for formatting
            max_taxon_length = max(len(str(taxon)) for taxon in binary_df.index)
            for taxon, seq in zip(binary_df.index, binary_df.values):
                tax_label = str(taxon).replace(" ", "_")
                f.write(f"{tax_label.ljust(max_taxon_length)}    {''.join(seq)}\n")
                # f.write(f"{tax_label}    {''.join(seq)}\n")
            f.write(";\nEnd;\n")

        # write mapping CSV
        mapping_df = pd.DataFrame(mapping_rows, columns=["column", "label_0", "label_1", "label_error"])
        # mapping_df.to_csv(mapping_path, index=False, sep=";", encoding="utf-8")
        # rename "column" to "character", delete "label_error" column
        mapping_df = mapping_df.rename(columns={"column": "character"}).drop(columns=["label_error"])
        with open(mapping_path, "w", encoding="utf-8") as f:
            f.write("#This file maps labels (0|1) to their corresponding meaning for each character. label '?' means error or whole description missing for the character.\n")
            mapping_df.to_csv(f, index=False, sep=";")

        # write binary matrix CSV
        binary_df.to_csv(binary_path, sep=";", encoding="utf-8")

        return binary_df, mapping_df


if __name__ == "__main__":
    ## make sure all necessary directories/files exist
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    if not TXT_FILES_DIR.exists():
        raise FileNotFoundError(f"TXT files directory not found: {TXT_FILES_DIR}")
    
    ## get arguments (timestamp is optional, default True)
    import argparse
    argparser = argparse.ArgumentParser(description="Parse XML and generate character matrix for phylogenetic analysis.")
    argparser.add_argument('-t', "--timestamp", default="yes", help="Whether to include timestamp in output directory name.")
    args = argparser.parse_args()
    if args.timestamp.lower() in ["yes", "y", "true", "1"]:
        timestamp = True
    elif args.timestamp.lower() in ["no", "n", "false", "0"]:
        timestamp = False
    else:
        print(f"Warning: Unrecognized timestamp argument '{args.timestamp}'. Defaulting to True.")
        timestamp = True

    ## 1. Parse XML and extract DataFrame
    parser = PhyloXMLParser(timestamp=True)
    parser.get_df(out=parser.processpath / "characters.csv")
    print("Character table extracted and saved.")

    ## 2. Create filename mapping table
    parser.get_file_mapping()
    if parser.file_mapping is not None:
        parser.file_mapping.to_csv(parser.processpath / "filename_mapping.csv", index=False, sep=";")
        print("Filename mapping table saved.")

    ## 3. Build binary matrix and mapping, save as NEXUS and CSV
    binary_df, mapping_df = parser.get_binary_matrix_nexus(
        nexus_path=parser.resultpath / "characters.nex",
        mapping_path=parser.resultpath / "characters_mapping.csv",
        binary_path=parser.resultpath / "characters_binary.csv"
    )
    print("Binary matrix and mapping files saved.")
