import torch.nn as nn
from transformers import T5ForConditionalGeneration


class T5ModelForIR(nn.Module):
    def __init__(self, args):
        super(T5ModelForIR, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(args.pretrained_path)

    def get_embedding(self, inputs):

        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])

        return embeddings

    def forward(self, inputs, labels):

        outputs = self.model(input_ids=inputs["input_ids"],
                             attention_mask=inputs["attention_mask"],
                             labels=labels["input_ids"])

        return outputs

    def set_forward(self, embeddings, inputs, labels):

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=inputs['attention_mask'],
            labels=labels["input_ids"]
        )

        return outputs

    def generate(self, inputs):

        generated = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs['attention_mask'],
            max_length=2
        )

        return generated
