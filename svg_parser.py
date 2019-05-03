import ast
import requests
import json
import os
import re
import copy
# from collections import OrderedDict
from flask import Flask,request,jsonify
from werkzeug.utils import secure_filename

EXTENSIONS=['.json']
UPLOAD_FOLDER = './svgjson_files'
# url="http://35.188.227.39:8080/enhancer/chain/scorpiosvchain"
# api_url="http://35.188.227.39:8080/enhancer/chain/scorpiosvchain"
api_url='http://35.227.82.195:8080/enhancer/chain/scorpiosvchain2'
header={ "Content-Type": "application/pdf"}

broker_tc=["SIZE","size","Size","vessel size","Vessel Size","1Year","1 Year high","1 Year low","3 Years","1 Year","5 Years"]
broker_tc_rate=["1Year","1 Year","3 Years","size","vessel size","Size","SIZE","type","Type","TYPE","vessel type","Vessel Size","5 Years","Guide","GUIDE"]
time_char=["Ship","Ship Name","Vessel Name","VESSEL NAME","vessel name","Name","Vessel","VESSEL","Period/mos","PERIOD","Period"]
spot=["Vessel Name","VESSEL NAME","Name","Vessel","VESSEL","Laycan","laycan","lc","Lc","LC","L/C","LAYCAN","Rate","rate","RATE","Rate/Hire","Hire","hire","HIRE","price","Price","PRICE","LOAD PORT","Load Port","load port","LoadPort","LOAD","Load","Disc Port","DISCH"]
tonnage=["Vessel Name","VESSEL NAME","Name","Vessel","VESSEL","EXPORT/PORT","EXPORT/DATE","Port","PORT","Open Port","OPEN PORT","OPEN DATE","Open Date","OPEN","DATE","Open","Date"]
built=['BLT','blt','Blt','build','yr','year built','Year Built','year','built year','Built','built','BUILT','BLT','BUILD','Build','Built Year','Yr','YR']
vessels=['Vessel','vessel', 'ship name', 'name', 'subvessel', 'ship','vessel name','VESSEL NAME']
bramer_comments=['last cgo','LAST CGO','comments','comment','comments/provisions','yard/comments', 'notes','REMARK','Remark','remark','REMARKS','Remarks','remarks','position note','posn note']
comments=["comment","comments"]
owners=['owner','owners','Owners','OWNERS','Owner','OWNER']
hw_dwt=['dwt','DWT','Dwt']
charterer=['charterer','chrts','fleet','chrts','chrt']
vesselsize=['size','vessel size','Size','SIZE','type','Type','TYPE','vessel type','Vessel Size']
year_1=['1 year','year 1','Year 1','1 yr','1Year','1 Year']
year_2=['2yrs','year 2','2 yr']
year_3=['3 years','3yrs','year 3','3 Years']
year_5=['5 years','5 Years','year 5','5yrs']
hw_delivery_loc=['delivery location','delivery place']
br_delivery_loc=['laycan','delivery']
hw_rate=['hire','Hire','rate','price','Price','PRICE','ratetype','rate type','Rate','RATE','Rate/Hire','Hire','hire','HIRE']
rates=['hire','rate','price']
period=['period','period/mos',"Period","PERIOD"]
status=['employement status','status','SUBS/HOLD','Subs/Hold']
cargo_qty=['qty','cargoqty','quantity','cargo qty']	
cargo_GDE=['gde','cargograde','grade','cargo grade']
cargo_type=['cargo type']
load_port=['load','load port','LOAD','Load',"Load Port"]
disch_port=['disch','discport','Discport','dicharge','Dicharge','disc','Disc','disch.','Disch.','disc port','Disch','DISCH']
open_port=['Open Port','OPEN PORT','openport','open port','port','PORT','OPEN','open','Open','OpenPort','EXPORT/DATE','EXPORT/PORT','export/date','export/port']
lc_start_end=['lc','start/end','l/c','lc end','lc start']
cubic=['cbm','cubic','cub','cbs','CBM','CUBIC','CUB','CBS','Cbm','Cubic','Cub','Cbs']
eta_basis=['ETA FUJAIRAH','eta fujairah','eta basis','ETA BASIS','eta fuj','eta','ETA','Eta','ETA FUJ','ETA SKO','eta sko','Eta Sko','KOREA','korea','Korea','ETA KOREA','eta korea']
reposition_reg=['position','reposition region']
open_date=['open date','Open Date','Date Open','date','date open','DATE','Date']
operator_list=["controlled by","operators","operator","controller","Controlled by","Operators","Operator","Controller","Fleet","FLEET","fleet"]
ice=['ICE/IMO','ice','Ice','ICE']

urlhash=r'(\#[A-Za-z].*)'

app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
	os.mkdir(UPLOAD_FOLDER)
	
def allowed_file(filename):
	filenm=os.path.basename(filename)
	if filenm:
		if os.path.splitext(filenm):
			ext=os.path.splitext(filenm)[1]
			if ext in EXTENSIONS:
				return True
	return False
	
@app.route("/svgjson",methods=["GET"])
def replace_urls():
	svgfile=request.args.get('file')
	data_returnedlist={}
	data={}
	with open(svgfile) as f:
		data = json.load(f)
			
	data_returnedlist=simplify_json(data)
	return jsonify(data_returnedlist)

def simplify_json(data2):
	simplified_json={}
	simplified_json_list=[]
	xsorted_simplified_json_list=[]
	ysorted_simplified_json_list=[]
	final_json={}
	x_coor=[]
	y_coor=[]
	sorted_x_coor=[]
	sorted_y_coor=[]
	flag=False
	list_from_rows_json={}
	list_from_cols_json=[]
	data=sorted(data2, key = lambda i: i["position"]["y"])
	if isinstance(data,list):
		for item in data:
			simplified_json={}
			flag=False
			if "text" in item.keys() and "position" in item.keys():
				if bool(item["text"].strip()):
					simplified_json.update({"text":item["text"].strip(),"x":item["position"]["x"],"y":item["position"]["y"]})
					if int(item["position"]["x"]) not in x_coor:x_coor.append(int(item["position"]["x"]))
					if int(item["position"]["y"]) not in y_coor:
						for i in y_coor:
							if int(item["position"]["y"])-i in range(0,3):
								flag=True
								# print "dd---",(int(item["position"]["y"]),i)
						if not flag:
							y_coor.append(int(item["position"]["y"]))
						
			if simplified_json:		
				# print "sj---",simplified_json
				simplified_json_list.append(simplified_json)
							
	sorted_y_coor=sorted(y_coor)
	ysorted_simplified_json_list=sorted(simplified_json_list, key = lambda i: i['y'])
	# data arranged row wise 
	list_from_rows_json=rows_from_json(ysorted_simplified_json_list,sorted_y_coor)
	# arrange row wise data in 3 parts
	final_json=three_parts_data(list_from_rows_json)
	
	# return list_from_rows_json
	return final_json
	
def rows_from_json(list_from_rows_json,sorted_y_coor):
	row_list=[]
	row_list2=[]
	all_rows=[]
	temp_all_rows=[]
	all_rows2=[]
	counter=1
	temp_json=list_from_rows_json
	
	for i in sorted_y_coor:
		row_list=[]
		row_list2=[]
		for item in temp_json:
			if int(item["y"])==i or (int(item["y"])-i) in range (3,-3,-1):
				row_list.append(item)
				row_list2.append(item["text"])
		# print "{0}-{1}".format(counter,row_list2)		
		counter+=1
		temp_all_rows.append(sorted(row_list, key = lambda i: i['x']))
		all_rows2.append(row_list2)
	
	for x,y in enumerate(temp_all_rows):
		for i,j in enumerate(y):
			if i>1 and i<len(y)-1:
				if "text" in j.keys():
					if j["text"] in ["+","-","_"]:		
						# print "\ns===",y[i-1]["text"]+y[i]["text"]+y[i+1]["text"]
						if "text" in y[i-1].keys() and "text" in y[i].keys() and "text" in y[i+1].keys():
							textval=y[i-1]["text"]+y[i]["text"]+y[i+1]["text"]
							y.append({"text":textval,"x":y[i-1]["x"],"y":y[i-1]["y"]})
							y[i-1]["text_used"]=y[i-1].pop("text","text_used_")
							y[i]["text_used"]=y[i].pop("text","text_used_")
							y[i+1]["text_used"]=y[i+1].pop("text","text_used_")
		all_rows.append(sorted(y, key = lambda i: i['x']))
					
	return all_rows	
	
def three_parts_data(list_from_rows_json):
	# header_data_withxy_list=[]
	# data_withxy_list=[]
	# three_parts_json=OrderedDict() 
	headers_count=0
	header_index=-1
	counter=1
	tables_dic={}
	table_data=[]
	row_data={}
	main_row_data={}
	row_data_list=[]
	headers=[]
	headers_dic={}
	table_data_dic={}
	table_data_list=[]
	tmp_headers_dic={}
	label_string=""
	all_tables_dic={}
	tcounter=0
	headers_flag=False
	header_data_list=[]
	for index,itemval in enumerate(list_from_rows_json):
		row=[i['text'] for i in itemval if 'text' in i.keys()]
		# print ('row----'," ".join(row).encode('utf-8'))
		if any(ele in row for ele in vessels):
			headers_count+=1
		if any(ele in row for ele in built):
			headers_count+=1
		if any(ele in row for ele in bramer_comments):
			headers_count+=1
		if any(ele in row for ele in hw_dwt):
			headers_count+=1
		if  any(ele in row for ele in charterer):
			headers_count+=1
		if  any(ele in row for ele in vesselsize):
			headers_count+=1
		if  any(ele in row for ele in year_1):
			headers_count+=1
		if  any(ele in row for ele in year_3):
			headers_count+=1
		if  any(ele in row for ele in year_5):
			headers_count+=1
		if  any(ele in row for ele in hw_delivery_loc):
			headers_count+=1
		if  any(ele in row for ele in hw_rate):
			headers_count+=1
		if  any(ele in row for ele in rates):
			headers_count+=1
		if  any(ele in row for ele in period):
			headers_count+=1
		if  any(ele in row for ele in br_delivery_loc):
			headers_count+=1
		if  any(ele in row for ele in status):
			headers_count+=1
		if  any(ele in row for ele in owners):
			headers_count+=1
		if  any(ele in row for ele in eta_basis):
			headers_count+=1
		if  any(ele in row for ele in ice):
			headers_count+=1
		if  any(ele in row for ele in open_date):
			headers_count+=1
		if  any(ele in row for ele in disch_port):
			headers_count+=1
		if  any(ele in row for ele in cargo_GDE):
			headers_count+=1
		if  any(ele in row for ele in cargo_qty):
			headers_count+=1
		if  any(ele in row for ele in cargo_type):
			headers_count+=1
		if  any(ele in row for ele in load_port):
			headers_count+=1
		if  any(ele in row for ele in open_port):
			headers_count+=1
		if  any(ele in row for ele in cubic):
			headers_count+=1	
		if headers_count>2 and not headers:
			headers=row
			headers_flag=True
			header_index=index
			if headers:
				label_string=" ".join(headers)
				for ind,item in enumerate(headers):
					headers_dic.update({ind:item})
				print ('headers----',headers)		
		# header part process
		if header_index==-1:
			print ('headerspart----'," ".join(row).encode('ascii','ignore'))		
			urls_list=""
			final_urls_list=[]
			stanboldic_r_copy=""
			stanboldic_r=""
			keyreplaced=""
			b_replace=""
			st=""
			header_row_data={}
			header_row_data_list=[]
			main_header_row_data={}
			urls_list=[]
			r = requests.post(url=api_url,headers=header,data=" ".join(row).encode('ascii','ignore'))
			
			if r:
				stanboldic_r = ast.literal_eval(r.text)
				stanboldic_r_copy=stanboldic_r
			if isinstance(stanboldic_r,list):
				# main_header_row_data={}
				for item in stanboldic_r:
					# header_row_data={}
					if isinstance(item,dict):
						for key,val in item.items():
							if key=="@type":
								if "http://fise.iks-project.eu/ontology/EntityAnnotation" in val:
									s=item['http://fise.iks-project.eu/ontology/entity-label']
									if isinstance(s,list):
										for l in s:
											if isinstance(l,dict):
												for a,b in l.items():
													if a=="@value":
														# if b.lower().strip()==i.lower().strip():
														z=item["http://fise.iks-project.eu/ontology/entity-reference"]
														id=item['http://purl.org/dc/terms/relation']
														header_row_data={}
														if z:
															if isinstance(z,list):
																tmpdict=z[0]
																if isinstance(tmpdict,dict):
																	for q,r in tmpdict.items():
																		if q=="@id":
																			keyreplaced=r
																			# print ("krepl---",keyreplaced)
																			
															
														if isinstance(id,list):
															for l in id:
																if isinstance(l,dict):
																	for a,bd in l.items():
																		if a=="@id":
																			digitvalue=bd
																			for item in stanboldic_r_copy:
																				if isinstance(item,dict):
																					for key,val in item.items():
																						if key=="@id":
																							if val==digitvalue:
																								d=item["http://fise.iks-project.eu/ontology/selected-text"]
																								# elif "http://fise.iks-project.eu/ontology/TextAnnotation" in val:
																								if isinstance(d,list):
																									# print "k--",d
																									for l in d:
																										if isinstance(l,dict):
																											for a,b in l.items():
																												if a=="@value":																										
																													# print ('word---',b)
																													# print ('url---',keyreplaced)
																													if keyreplaced!="":
																														urlpatt=re.findall(urlhash,keyreplaced)
																														if urlpatt:
																															if urlpatt[0].lower() not in urls_list:
																																header_row_data.update({'url':keyreplaced,'word':b})
																																urls_list.append(urlpatt[0].lower())
																															# print ('ulist----',urls_list)
																															if header_row_data:
																																header_row_data_list.append(header_row_data)
																																# break
																													
				if header_row_data_list:																									
					main_header_row_data.update({'data':header_row_data_list,'line_index':index})																									
					
				if main_header_row_data:
					header_data_list.append(main_header_row_data)				
		# ==========process table part========================================
		if headers_dic and index>header_index:
			# make table json
			row_data={}
			main_row_data={}
			table_data_dic={}
			tmp_headers_dic=copy.deepcopy(headers_dic)
			for id,im in enumerate(row):
				row_data.update({id:im})
	
			if row_data:
				# if len(row_data.keys())==len(tmp_headers_dic.keys()):
					# # main_row_data.update({'data':row_data,'header':tmp_headers_dic})
					# # print ('equal---',row_data,tmp_headers_dic)
				if len(row_data.keys())<len(tmp_headers_dic.keys()):
					for k,v in enumerate(tmp_headers_dic.keys()):
						if v not in row_data.keys():					 
							row_data.update({v:""})
				elif len(row_data.keys())>len(tmp_headers_dic.keys()):
					counter=1
					for k,v in enumerate(row_data.keys()):
						if v not in tmp_headers_dic.keys():
							tmp_headers_dic.update({v:'unknown_'+str(counter)})	
							counter+=1
				main_row_data.update({'data':row_data,'header':tmp_headers_dic,'label_line_index':header_index,'label_string':label_string,'line_index':index})
						
			if main_row_data:
				table_data_list.append(main_row_data)

			if table_data_list:
				table_data_dic.update({'table_data':table_data_list})
			
			if header_data_list:
				# header_data_list.append(main_header_row_data)
				table_data_dic.update({'header_data':header_data_list})
				
		if headers_flag==True:
			if table_data_dic:
				all_tables_dic.update({'table_'+str(tcounter):table_data_dic})
				tcounter+=1
				headers_flag=False
			
	
	return all_tables_dic
	
if __name__=="__main__":
	app.run(debug=True,host="0.0.0.0",port=5035)
