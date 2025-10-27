
using CSV,DataFrames


df=CSV.read("./SMU_UA159/QC Analysis/Golden_Gate_Mantis/RapidKO_Rework_Golden_Gate_Map.csv",DataFrame)
df=CSV.read("./SMU_UA159/QC Analysis/Golden_Gate_Mantis/OpMod_Rework_Golden_Gate_Map.csv",DataFrame)


plates=49:50

data=df[map(x->in(x,plates),df.Plate_ID),:]

totalvols=sum.(eachcol(data))[1:end-1]

colnames=names(data)[1:end-1]

prepvols=map(x->max(x+25,x*1.25),totalvols)

outvols=map(x->ifelse(x<=25,0,x),prepvols)

outdf=DataFrame(outvols',colnames)

CSV.write("/Users/BDavid/Desktop/golden_gates_49_50.csv",outdf)