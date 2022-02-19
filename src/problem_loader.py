import os
import pickle


class ProblemLoader(object):
    def __init__(self, problems_dir):
        self.problems_dir = problems_dir
        self.problems_list = os.listdir(problems_dir)
        self.next_pointer = 0
        self.problems_by_files = self.load_problems()

    def has_next(self):
        return self.next_pointer < len(self.problems_by_files)

    def load_problems(self):
        problems_by_files = []
        for file_idx in range(len(self.problems_list)):
            file_name = self.problems_list[file_idx]
            file_name = os.path.join(self.problems_dir, file_name)
            with open(file_name, 'rb') as f:
                problems = pickle.load(f)
                problems_by_files.append(problems)

        return problems_by_files

    def get_next(self):
        if not self.has_next():
            self.reset()
        problems = self.problems_by_files[self.next_pointer]
        self.next_pointer += 1
        return problems

    def reset(self):
        self.next_pointer = 0
