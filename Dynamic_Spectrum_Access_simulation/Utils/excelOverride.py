import xlrd
import openpyxl
import time

start_time = time.time()
filename = "./tx.xlsx"
workbook1 = xlrd.open_workbook(filename=filename)
sheet = workbook1.sheet_by_index(0)
row_num = len(sheet.col_values(1))

w_workbook = openpyxl.load_workbook(filename)
w_sheet = w_workbook["memory_store"]

col_next = sheet.row_values(1)
zero_loc = col_next.index(0)

for i in range(2, row_num):

    col = col_next[:zero_loc]
    col_next = sheet.row_values(i)[:zero_loc]
    print("当前行", i)
    for index, value in enumerate(col_next):
        if value == '':
            # w_sheet.write(i+1, index, col[index])
            w_sheet.cell(row=(i+1), column=(index+1)).value = col[index]    # !!!
            col_next[index] = col[index]
sav_start_time = time.time()
w_workbook.save("./tx.xlsx")
end_time = time.time()
print("存储时间：", end_time - sav_start_time)
print("总时间：", end_time - start_time)