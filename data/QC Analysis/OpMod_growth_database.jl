using DataFrames, CSV, XLSX 


SMU_data=CSV.read("./SMU_UA159/SMU_UA159_Operon_Transcriptional_Modulation_Oligo_Data_9723_Final.csv",DataFrame)

enzymes=unique(SMU_data[:,:Golden_Gate_Enzyme])

enzyme_data=String.(SMU_data[:,:Golden_Gate_Enzyme])


strand_data=String.(SMU_data[:,:Strand])
strand_data=strand_data.=="+" # 0 for - strand, 1 for + strand

ku_payloads=["KU_Payload_$i" for i in 1:2*length(enzymes)]
kd_payloads=["KD_Payload_$i" for i in 1:2*length(enzymes)]

enzyme_dict=Dict(enzymes .=> 1:2:2*length(enzymes))

ku_payload_data=ku_payloads[map(x->enzyme_dict[x],enzyme_data).+strand_data]
kd_payload_data=kd_payloads[map(x->enzyme_dict[x],enzyme_data).+strand_data]

growth_database=vcat(SMU_data,SMU_data)
growth_database[:,:Payload_Name].=vcat(ku_payload_data,kd_payload_data)

n=nrow(SMU_data)

growth_database[1:n,:Plate_ID] .= SMU_data[:,:Plate_ID] .+26
growth_database[n+1:end,:Plate_ID].= SMU_data[:,:Plate_ID] .+35

ids=CSV.read("./SMU_UA159/Prior Growth Data/SMU_UA159_identifiers.csv",DataFrame)


growth_database=growth_database[:,[:Plate_ID,:Row,:Col,:Gene_ID,:Golden_Gate_Enzyme,:Payload_Name,:PTag_1_ID,:PTag_2_ID]]
k=ncol(growth_database)
id_codes=names(ids)
l=length(id_codes)
n_rows=nrow(growth_database)
default_ids=["" for i in 1:n_rows,j in 1:length(id_codes)]
default_id_table=DataFrame(default_ids,id_codes)
growth_database=hcat(growth_database,default_id_table)

for row in 1:nrow(growth_database)
    gene_id=growth_database[row,:GeneID]
    query=join(split(gene_id,"_"),".") # in house id uses "_" , while quivey uses "." 
    ids_row=findfirst(x->x==query,ids[:,3])

    if typeof(ids_row)==Nothing
        continue 
    else 
        growth_database[row,7:11].=Vector(ids[ids_row,:])
    end 
end 


### add our production growth data 


colony_growth=["" for i in 1:nrow(growth_database)]
liquid_growth=["" for i in 1:nrow(growth_database)]
start_plate=27
end_plate=44
for plate in start_plate:end_plate

    plate_data=XLSX.readxlsx("./SMU_UA159/Growth Summary/plate$plate.xlsx")
    col_names=plate_data[1][:][1,:]
    data=plate_data[1][:][2:end,:]
    plate_data_df=DataFrame(data,col_names)
    start_idx=(96*(plate-start_plate)+1)
    colony_growth[start_idx:start_idx+nrow(plate_data_df)-1].=plate_data_df[:,:Plate]
    liquid_growth[start_idx:start_idx+nrow(plate_data_df)-1].=plate_data_df[:,:Liquid1]



end 


our_production=DataFrame(colony_growth=colony_growth,liquid_growth=liquid_growth)
growth_database=hcat(growth_database,our_production)

# mutant growth 
growthcurves=CSV.read("./SMU_UA159/QC Analysis/mutant_growth_curve_data.csv",DataFrame)

growthcurves=subset(growthcurves,:PlateID => a -> 27 .<= a .<= 44 )
wellid=growthcurves[:,:WellID]

splitwells=map(x->(string(x[1]),Meta.parse(String(x[2:end]))),wellid)
Row= String.(map(x->x[1],splitwells))
Col= Int64.(map(x->x[2],splitwells))
growthcurves[:,:Row].=Row
growthcurves[:,:Col].=Col
sort!(growthcurves,[:PlateID,:Col,:Row])
growthcurves=growthcurves[1:nrow(growth_database),:]
growth_database=hcat(growth_database,growthcurves[:,[:RawMaxOD,:TimeMaxOD,:PredVMax,:TimePredVMax]])




CSV.write("./SMU_UA159/QC Analysis/OpMod_growth_database.csv",growth_database)