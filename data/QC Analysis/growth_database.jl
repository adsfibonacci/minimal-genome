using DataFrames, CSV, XLSX ,Statistics, LinearAlgebra


growth_master=CSV.read("./SMU_UA159/SMU_UA159_RapidKO_Master_82523.csv",DataFrame)

splitwells=map(x->(string(x[1]),Meta.parse(String(x[2:end]))),growth_master[:,:Well_ID])
Row= String.(map(x->x[1],splitwells))
Col= Int64.(map(x->x[2],splitwells))
growth_master[:,:Row].=Row
growth_master[:,:Col].=Col
ids=CSV.read("./SMU_UA159/Prior Growth Data/SMU_UA159_identifiers.csv",DataFrame)


growth_database=growth_master[:,[:Plate_ID,:Row,:Col,:New_ID,:Golden_Gate_Enzyme,:Payload_Name,:PTag_1_ID,:PTag_2_ID]]
k=ncol(growth_database)
id_codes=names(ids)
l=length(id_codes)
n_rows=nrow(growth_database)
default_ids=["" for i in 1:n_rows,j in 1:length(id_codes)]
default_id_table=DataFrame(default_ids,id_codes)
growth_database=hcat(growth_database,default_id_table)

for row in 1:nrow(growth_database)
    gene_id=growth_database[row,:New_ID]
    query=join(split(gene_id,"_"),".") # in house id uses "_" , while quivey uses "." 
    ids_row=findfirst(x->x==query,ids[:,3])

    if typeof(ids_row)==Nothing
        continue 
    else 
        growth_database[row,(k+1):(k+l)].=Vector(ids[ids_row,:])
    end 
end 

col_names=names(growth_database)

col_names[k+3]="Locus"
rename!(growth_database,col_names)

### add our production growth data 


colony_growth=["" for i in 1:nrow(growth_database)]
liquid_growth=["" for i in 1:nrow(growth_database)]
n_plates=26#26 ############################################# FIX THIS ###############
for plate in 1:n_plates 

    plate_data=XLSX.readxlsx("./SMU_UA159/Growth Summary/plate$plate.xlsx")
    col_names=plate_data[1][:][1,:]
    data=plate_data[1][:][2:end,:]
    plate_data_df=DataFrame(data,col_names)
    start_idx=(96*(plate-1)+1)
    colony_growth[start_idx:start_idx+nrow(plate_data_df)-1].=plate_data_df[:,:Plate]
    liquid_growth[start_idx:start_idx+nrow(plate_data_df)-1].=plate_data_df[:,:Liquid1]



end 


our_production=DataFrame(colony_growth=colony_growth,liquid_growth=liquid_growth)
growth_database=hcat(growth_database,our_production)


### Quivey data 

quivey_successful=["" for i in 1:nrow(growth_database)]
quivey_essential=["" for i in 1:nrow(growth_database)]

quivey_db_path="./SMU_UA159/Prior Growth Data/Quivey SMU UA159 Deletion Mutant Database.xlsx"
quivey_db=XLSX.readxlsx(quivey_db_path)
col_names=quivey_db[1][:][1,[3,1]]
col_names[2]="Mutant"
data=quivey_db[1][:][2:1963,[3,1]]
quivey_df=DataFrame(data,col_names)
deleteat!(quivey_df, findall(x->typeof(x)==Missing,quivey_df[:,:Locus]))
for row in 1:nrow(growth_database)
    gene_id=growth_database[row,:New_ID]
    query=join(split(gene_id,"_"),".") # in house id uses "_" , while quivey uses "." 
    quivey_row=findfirst(x->x==query,quivey_df[:,:Locus])

    if typeof(quivey_row)==Nothing
        continue 
    else 

        code=quivey_df[quivey_row,2]
        if code=="y" 
            quivey_successful[row]="Y"
            quivey_essential[row]="N"
        elseif code=="n"
            quivey_essential[row]="Y"
            quivey_successful[row]="N"
        else 
            quivey_successful[row]="N"
            quivey_essential[row]="?"
        end 
    end 
end 
quivey_data=DataFrame(quivey_successful=quivey_successful,quivey_essential=quivey_essential)
growth_database=hcat(growth_database,quivey_data)


# binarize quivey data for essentiality 

rows=findall(x->x=="y"||x=="n",quivey_df[:,:Mutant])

essential_df=quivey_df[rows,:]
essential_df[!,:essential].=map(x->x=="n",essential_df[:,:Mutant])
essential_df[!,:Locus].=map(x->join(split(x,"."),"_"),essential_df[!,:Locus])


### Shields essential genes data 

shields_path="./SMU_UA159/Prior Growth Data/Shields Tn-seq Counts.xlsx"

shields_tb=XLSX.readxlsx(shields_path)
colnames=shields_tb[1][:][2,[1,3,4,5,6]]
colnames[1]="Locus"
data=shields_tb[1][:][3:end,[1,3,4,5,6]]
shields=DataFrame(data,colnames)
shields_omit_path="SMU_UA159/Prior Growth Data/Shields Table S1 - Duplicated and small genes.xlsx"
shields_omit_tb=XLSX.readxlsx(shields_omit_path)
shields_dup=shields_omit_tb[1][:][3:43,1] # duplicates that are expected to be nonessential but may show up as essential in tn-seq counts 
shields_omit=shields_omit_tb[1][:][45:113,1] # small genes that are difficult to detect using tn-seq method. They may or may not be essential 
for locus in shields_omit
    deleteat!(shields,findall(x->x==locus,shields[:,:Locus]))
end 

in_both=intersect(essential_df[!,:Locus],shields[!,:Locus])

quivey_essential=[false for i in 1:length(in_both)]
for i in eachindex(in_both)
    row = findfirst(x->x==in_both[i],essential_df[!,:Locus])
    quivey_essential[i]=essential_df[row,:essential]
end 
function cos_similarity(a::Vector{Bool},b::Vector{Bool})

    return dot(a,b)/norm(a)/norm(b)
end 


mean_counts=mean(Array(shields[:,2:end]),dims=2)

log_mean_counts=log.(mean_counts)


thresholds=0:0.01:4
sim_scores=zeros(length(thresholds))

threshold=2.5 
essential=log_mean_counts .<threshold
shields[!,:essential].=essential
for locus in shields_dup
    row = findfirst(x->x==locus,shields[:,:Locus])
    if typeof(row)==Nothing
        continue 
    end 

    shields[row,:essential]=false
end 

shields_essential=[false for i in 1:length(in_both)]
for i in eachindex(in_both)
    row = findfirst(x->x==in_both[i],shields[!,:Locus])
    shields_essential[i]=shields[row,:essential]
end 

cos_similarity(quivey_essential,shields_essential)





####### MUTANT GROWTH DATA 
growthcurves=CSV.read("./SMU_UA159/QC Analysis/mutant_growth_curve_data.csv",DataFrame)

growthcurves=subset(growthcurves,:PlateID => a -> 1 .<= a .<= 26 )
wellid=growthcurves[:,:WellID]

splitwells=map(x->(string(x[1]),Meta.parse(String(x[2:end]))),wellid)
Row= String.(map(x->x[1],splitwells))
Col= Int64.(map(x->x[2],splitwells))
growthcurves[:,:Row].=Row
growthcurves[:,:Col].=Col
sort!(growthcurves,[:PlateID,:Col,:Row])
growthcurves=growthcurves[1:nrow(growth_database),:]

growth_database=hcat(growth_database,growthcurves[:,[:RawMaxOD,:TimeMaxOD,:PredVMax,:TimePredVMax]])
CSV.write("./SMU_UA159/QC Analysis/growth_database.csv",growth_database)

#=
maxOD_threshold=0.6

tmaxOD_threshold= 9

vmax_threshold=0.2

t_pred_vmax_lower=5
t_pred_vmax_upper=15 
=#








    





