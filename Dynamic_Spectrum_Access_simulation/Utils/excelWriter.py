import xlsxwriter


class dataWriter:

    def __init__(self, sheet_name='train_data', file_name='train_data.xlsx'):
        self.workbook = xlsxwriter.Workbook(file_name)
        # self.worksheet = self.workbook.add_worksheet(sheet_name)
        self.file_name = file_name
        self.column_map = {}
        self.current_column_num = 0

    def create_sheet(self, sheet_name):
        self.worksheet = self.workbook.add_worksheet(sheet_name)
        self.current_column_num = 0

    def close(self):
        # self.workbook.save(self.file_name)
        self.workbook.close()

    def write_into_lable(self, rows, data, label='success_rate'):
        if label not in self.column_map.keys():
            # print(label)
            self.column_map[label] = self.current_column_num
            self.current_column_num += 1
            # print(self.column_map[label])
            self.worksheet.write(0, self.column_map[label], label)
        #     print(label, " and ", 0)

        if isinstance(data, str):
            self.worksheet.write(rows, self.column_map[label], data)
        else:
            # print(type(data) == '<class \'str\'>', type(data)=='str')
            self.worksheet.write(rows, self.column_map[label], float(data))
        # print(data, " and ", rows)




