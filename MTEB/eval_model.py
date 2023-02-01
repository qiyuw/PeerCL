import torch
import argparse
import numpy as np
from mteb import MTEB
from tqdm import trange
# refer https://github.com/embeddings-benchmark/mteb#leaderboard for the usage of MTEB.
from transformers import AutoModel, AutoTokenizer

class PCLforMTEB():
    def __init__(self, model_name):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        num_steps = len(sentences) // batch_size
        if len(sentences) % batch_size > 0:
            num_steps += 1
        embedding_list = []
        for step in trange(0, num_steps, desc='Encoding'):
            sent_batch = sentences[step*batch_size:(step+1)*batch_size]
            inputs = self.tokenizer(sent_batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            embedding_list.append(embeddings.detach().cpu().numpy())
        embeddings_all = np.concatenate(embedding_list, axis=0)
        assert embeddings_all.shape[0] == len(sentences)
        return embeddings_all

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="name of PCL model")
    parser.add_argument("--bz", type=int, default=32, help="batch size for encoding")
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
    # evaluation.run(model, eval_splits=["test"], output_folder=f"results/{args.model}")
    evaluation.run(model, output_folder=f"results/{args.model}", batch_size=args.bz)