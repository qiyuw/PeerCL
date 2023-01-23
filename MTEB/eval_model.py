import torch
import argparse
from mteb import MTEB
# refer https://github.com/embeddings-benchmark/mteb#leaderboard for the usage of MTEB.
from transformers import AutoModel, AutoTokenizer

class PCLforMTEB():
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        return embeddings

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="name of PCL model")
    parser.add_argument("--tasks", nargs='+', default=None, help="task name in MTEB, see https://github.com/embeddings-benchmark/mteb#leaderboard.")
    parser.add_argument("--task-types", nargs='+', default=None, help="task name in MTEB, see https://github.com/embeddings-benchmark/mteb#leaderboard.")
    parser.add_argument("--task-langs", nargs='+', default=['en'], help="task name in MTEB, see https://github.com/embeddings-benchmark/mteb#leaderboard.")
    args = parser.parse_args()
    tasks = args.tasks
    task_types = args.task_types
    task_langs = args.task_langs
    model = PCLforMTEB(args.model)
    assert (task_types or tasks) is not None, "either task_types or tasks is required."
    if task_types is not None:
        evaluation = MTEB(task_types=task_types, task_langs=task_langs)
    else:
        evaluation = MTEB(tasks=tasks, task_langs=task_langs)
    evaluation.run(model, eval_splits=["test"])