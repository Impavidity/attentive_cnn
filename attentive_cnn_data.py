from torchtext import data
import os


class SimpleQADataset(data.TabularDataset):
    dirname = 'data'
    @classmethod
    def splits(cls, id_field, word_field, char_field, score_field, label_field,
               train='attentivecnn.train', validation='attentivecnn.valid', test='attentivecnn.test'):
        prefix_name = ''
        path = './data'
        return super(SimpleQADataset, cls).splits(
            os.path.join(path, prefix_name), train, validation, test,
            format='TSV', fields=[('id1', id_field), ('pattern1', word_field), ('mention1', char_field), ('candidate1', char_field), \
                                  ('predicate1', word_field), ('score1', score_field), ('label1', label_field), \
                                  ('id2', id_field), ('pattern2', word_field), ('mention2', char_field), ('candidate2', char_field), \
                                  ('predicate2', word_field), ('score2', score_field), ('label2', label_field)]
        )