using CSV, DataFrames,Plots,Distributions
# RapidKO Rework 

function sample_rows(df::DataFrame,n=1)
    k=nrow(df)

    return df[sample(1:k,n;replace=false),:]
end 


growth=CSV.read("./SMU_UA159/QC Analysis/growth_database.csv",DataFrame)


maxOD_threshold=0.25

tmaxOD_threshold= 7

vmax_lower=0.1
vmax_upper=0.6

t_pred_vmax_lower=5
t_pred_vmax_upper=20

histogram(growth[:,:RawMaxOD]);
xlabel!("Maximum OD");
vline!([maxOD_threshold])
histogram(growth[:,:TimeMaxOD]);
xlabel!("Time to Reach Maximum OD (hours)");
vline!([tmaxOD_threshold])
histogram(growth[:,:PredVMax]);
xlabel!("Predicted Maximum Growth Rate (OD/hour)");
vline!([vmax_lower,vmax_upper])
histogram(growth[:,:TimePredVMax]);
xlabel!("Predicted Time to Reach Maximum Growth Rate (hours)");
vline!([t_pred_vmax_lower,t_pred_vmax_upper])


growth[:,:PassColony]=growth[:,:colony_growth] .== "Y"
growth[:,:PassLiquid]=growth[:,:liquid_growth] .== "Y"
growth[:,:PassMaxOD]=growth[:,:RawMaxOD] .>= maxOD_threshold
growth[:,:PassTimeMaxOD]=growth[:,:TimeMaxOD] .>= tmaxOD_threshold
growth[:,:PassVMax]= vmax_lower .<= growth[:,:PredVMax] .<= vmax_upper
growth[:,:PassTimeVMax]= t_pred_vmax_lower .<= growth[:,:TimePredVMax] .<=t_pred_vmax_upper

a=growth[:,:PassColony]
b=growth[:,:PassLiquid]
c=growth[:,:PassMaxOD]
d=growth[:,:PassTimeMaxOD]
e=growth[:,:PassVMax]
f=growth[:,:PassTimeVMax]
growth[:,:PassAll] = all.(map((t,u,v,w,x,y)->[t,u,v,w,x,y],a,b,c,d,e,f))
growth[:,:Pass5] = sum.(map((t,u,v,w,x,y)->[t,u,v,w,x,y],a,b,c,d,e,f)) .>=5
growth[:,:Pass4] = sum.(map((t,u,v,w,x,y)->[t,u,v,w,x,y],a,b,c,d,e,f)) .>=4
growth[:,:Pass3] = sum.(map((t,u,v,w,x,y)->[t,u,v,w,x,y],a,b,c,d,e,f)) .>=4
CSV.write("./SMU_UA159/QC Analysis/growth_database.csv",growth)
sum(growth[:,:PassAll])
sum(growth[:,:Pass5])
sum(growth[:,:Pass4])
sum(growth[:,:Pass3])

growth[:,:Index]=collect(1:nrow(growth))


rework=subset(growth,:PassAll => x->x .==false )

rework[:,:IsRework] .=true 
pass=subset(growth,:PassAll => x->x .==true )

n=nrow(rework)

s=96
k=cld(n,s)
extras=min(6*k,s-mod(n,s)) # number of extra mutants to make, up to 6(number of plates)

per_plate=fld(extras,k)

interval=s-per_plate

cutoff_idxs=vcat(1,rework[[min(interval*x,n) for x in 1:k],:Index])

for i in 1:k
    passdf=subset(pass,:Index => x-> cutoff_idxs[i] .<= x .<= cutoff_idxs[i+1])
    passdf[:,:IsRework] .=false 
    posrows=sample_rows(passdf,per_plate)
    rework=vcat(rework,posrows)
end 

sort!(rework,:Index) 
start_plate= 45 

plate_ids = start_plate:(start_plate+k-1)
plate_rows=growth[1:s,:Row]
plate_cols= growth[1:s,:Col]

plate_id= vcat([[i for _ in 1:s] for i in plate_ids]...)
rows= vcat([plate_rows for _ in 1:k]...)
cols = vcat([plate_cols for _ in 1:k]...)

rework[:,:Plate_ID] .= plate_id[1:nrow(rework)]
rework[:,:Row] .= rows[1:nrow(rework)]
rework[:,:Col] .= cols[1:nrow(rework)]

CSV.write("./SMU_UA159/QC Analysis/rapidKO_rework.csv",rework)




scatter(1:nrow(rework),.!rework[:,:IsRework],legend=false);
vline!(vcat(1,[s*x for x in 1:k]))










#=
#######################################################
# Paul's crazy deletion Analysis

genes=subset(growth, :New_ID => x -> startswith.(x,"SMU"))


no_grows=vcat(0,findall(x->x==false,genes[:,:PassAll]))

contigs=zeros(Int64,length(no_grows)-1)
for n in 1:(length(no_grows)-1)
    contigs[n]=no_grows[n+1]-no_grows[n]-1
end 


    
sort!(contigs,rev=true)

out=vcat(1966,nrow(genes) .- cumsum(contigs))

out= out / nrow(genes)

plot(0:(length(out)-1),out,legend=false);

ylims!(0,1);
xlabel!("'Nonessential' Sections Removed");
ylabel!("Fraction of Genes Remaining")


#######################################
=# 

#=
9/6/24

Creating PTag plates from RapidKO_rework.csv

=# 

using JensenLabDispense

## Load in and process plate maps
data= CSV.read("./SMU_UA159/QC Analysis/RapidKO_rework.csv",DataFrame)

PTag_Map = CSV.read("./SMU_UA159/PTag_Plate_Maps.csv",DataFrame)

tag_names = String.(PTag_Map[:,:Sequence_Name])

in_plate1=reshape(permutedims(reshape(tag_names[1:96],12,8)),96,1)
in_plate2=reshape(permutedims(reshape(tag_names[97:end],12,8)),96,1)
PTag_Order=vcat(in_plate1,in_plate2)


## Parameter settings 

PTag_volume=10 # Dispense volume of each PTag primer 

experiment_name="SMU_UA159_RapidKO_Rework"
## Loop through each set of 96 experiments
plates=unique(data[:,:Plate_ID])
n_plates=length(unique(data[:,:Plate_ID]))
Initial_PTag_volume=[1000 for _ in 1:length(PTag_Order)]
remaining=zeros(n_plates+1,length(PTag_Order))
remaining[1,:]=Initial_PTag_volume
for i in eachindex(plates)
    Target_Map=data[data[:,:Plate_ID].==plates[i],:]

    design=zeros(96,length(PTag_Order))


    for tag in eachindex(PTag_Order)
        expts=union(findall(x->x==PTag_Order[tag],Target_Map[:,:PTag_1_ID]),findall(x->x==PTag_Order[tag],Target_Map[:,:PTag_2_ID]))
        design[expts,tag].=1
    end 

    design=PTag_volume*design
    total_used=vec(sum(1.1*design,dims=1))
    remaining[i+1,:]=remaining[i,:].-total_used

    df=DataFrame(design, :auto)

    plate_1_96=df[:,1:96]
    plate_97_192=df[:,97:192]


    directory1=string("./SMU_UA159/P_Tag_Dispensing/",experiment_name,"_Plate$(plates[i])","_PTag_1_96/")
    directory2=string("./SMU_UA159/P_Tag_Dispensing/",experiment_name,"_Plate$(plates[i])","_PTag_97_192/")

    source=labware["dWP96_2mL"]

    destination=labware["WP96"]

    liquidclasses=["Water" for _ in 1:96]

    #CobraDispense(plate_1_96,directory1,source,destination,liquidclasses;washtime=5000,dispensepause=true,predispensecount=0)
    #CobraDispense(plate_97_192,directory2,source,destination,liquidclasses;washtime=5000,dispensepause=true,predispensecount=0)


end 

min_remaining=minimum(remaining[end,:])
println("Minimum_remaining volume: $min_remaining")




### Golden Gates 


using DataFrames, CSV, JensenLabDispense


SMU_data = CSV.read("./SMU_UA159/QC Analysis/rapidKO_rework.csv",DataFrame)

enzymes=unique(SMU_data[:,:Golden_Gate_Enzyme])

enzyme_data=String.(SMU_data[:,:Golden_Gate_Enzyme])

payload=unique(SMU_data[:,:Payload_Name])

payload_data=String.(SMU_data[:,:Payload_Name])

n=nrow(SMU_data)
m=length(enzymes)
enzyme_volume=7.5 # 0.5 ul cutter, 0.5 ul T4 DNA lig , 1 ul T4 buffer, 5.5 ul H20 

design=zeros(n,m)

for i = 1:m
    x=enzyme_data .== enzymes[i]
    design[:,i]= enzyme_volume * x
end 

p=length(payload)
payload_design=zeros(n,p)

payload_volume=0.5

for i =1:p 
    y=payload_data .== payload[i]
    payload_design[:,i]=payload_volume * y
end 


enzyme_dispense=DataFrame(design,enzymes)

payload_dispense=DataFrame(payload_design,payload)

dispense=hcat(enzyme_dispense,payload_dispense)

dispense=hcat(dispense,SMU_data[:,:Plate_ID])

dispense=rename(dispense,vcat(enzymes,payload,"Plate_ID"))

CSV.write("./SMU_UA159/QC Analysis/RapidKO_Rework_Golden_Gate_Map.csv",dispense)


##### Make a dataframe for each plate that includes only the relevant columns 
Row=repeat(["A","B","C","D","E","F","G","H"],12)
Col=repeat(1:12, inner = 8)
IDX=DataFrame(Row=Row, Col=Col)

plates=unique(dispense[:,:Plate_ID])
currentplate=dispense
for i in plates

    plate=dispense[dispense[:,:Plate_ID].==i,1:12]
    colsums=[sum(plate[:,i]) for i= 1:12] .> 0 
    plate=plate[:,colsums]
    

    rows=nrow(plate)
    rowpad=96-rows
    colnames=names(plate)
    cols=length(colnames)
    pad=DataFrame(zeros(rowpad,cols),colnames)
    plate=vcat(plate,pad)
    plate=hcat(plate,IDX[1:96,:])
    currentplate=deepcopy(plate)
    CSV.write("./SMU_UA159/QC Analysis/Golden_Gate_Mantis/Plate_Dispense_Maps/SMU_UA159_Plate_$i.csv",plate)
    MantisDispense(plate[:,1:end-2],"./SMU_UA159/QC Analysis/Golden_Gate_Mantis/Plate_Dispense_Lists/SMU_UA159_Plate_$i",labware["brPCR96"])
end 


