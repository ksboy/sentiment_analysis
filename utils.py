import os, csv, json
import logging
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer
from transformers.data.processors import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)

class ChnSentiCorpProcessor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""
    def __init__(self):
        self.output_mode = 'classification'

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=",", quotechar=quotechar))

    # def get_example_from_tensor_dict(self, tensor_dict):
    #     """See base class."""
    #     return InputExample(
    #         tensor_dict["idx"].numpy(),
    #         tensor_dict["sentence"].numpy().decode("utf-8"),
    #         None,
    #         str(tensor_dict["label"].numpy()),
    #     )

    def get_train_examples(self, data_dir):
        """See base class."""
        result = []
        for file in [ "ChnSentiCorp_htl_all.csv", "weibo_senti_100k.csv", "waimai_10k.csv"]:
            examples = self._create_examples(self._read_csv(os.path.join(data_dir, file)), "train")
            for i in range(len(examples)):
                if i%5==0: continue
                else: result.append(examples[i])
        return result

    def get_dev_examples(self, data_dir):
        """See base class."""
        result = []
        for file in [ "ChnSentiCorp_htl_all.csv", "weibo_senti_100k.csv", "waimai_10k.csv"]:
            examples = self._create_examples(self._read_csv(os.path.join(data_dir, file)), "dev")[::5]
            result.extend(examples)
        return result

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            label = line[0]
            text_a = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    processor: DataProcessor = None,
    max_length: Optional[int] = None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if processor is not None:
        if label_list is None:
            label_list = processor.get_labels()
        if output_mode is None:
            output_mode = processor.output_mode

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float]:
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features

def write_file(datas, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")