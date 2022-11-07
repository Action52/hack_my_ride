list.files(".../gtfs3Sept")
routes = read.csv(".../routes.txt")
stops = read.csv(".../stops.txt")
trips = read.csv(".../trips.txt")
agency = read.csv(".../agency.txt")
stoptime = read.csv(".../stop_times.txt")
shapes = read.csv(".../shapes.txt")
calendar = read.csv(".../calendar.txt")


library(knitr)
kable(head(stops))
library(stringr)
library(tidyft)

# alternative installation of the %>%
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)
library(knitr)
kable(head(brussels_3sep))
library(ggplot2)
library(ggthemes)
library(rstudioapi)
register_google(key="Your key")
library(ggmap)

# Get coordinates for each stop
brussels_3sep <-(
  select(routes, route_id, route_short_name) %>% 
  inner_join(select(trips, route_id, trip_id)) %>% 
  inner_join(select(stoptime, trip_id, stop_id)) %>% 
  select(-trip_id) %>% unique() %>% 
  inner_join(select(stops, stop_id, stop_name, lat=stop_lat, lon=stop_lon)) %>% 
  unique())

# Get coordinates for each stop and counts of trips passing through each stop
  
brussels_3sep_cnt <- (
  select(routes, route_id,route_short_name) %>% 
    inner_join(select(trips, route_id, trip_id)) %>% 
    inner_join(select(stoptime, trip_id, stop_id)) %>% 
    group_by(stop_id) %>% summarise(cnt=n()) %>% 
    inner_join(select(stops, stop_id, stop_name, lat=stop_lat, lon=stop_lon)) %>% 
    unique())



#Plot the Transport Network
lx_map <- get_map(location = c(4.3514, 50.8533), maptype = "roadmap", zoom = 12)
# plot the map with a line for each group of shapes (route)
ggmap(lx_map, extent = "device") +
  geom_path(data = shapes, aes(shape_pt_lon, shape_pt_lat, group = shape_id), size = .1, alpha = .5, color='blue') +
  coord_equal() + theme_map()

#join all data and count number of services grouped by stop
stops_freq = 
  inner_join(stoptime,stops,by=c("stop_id")) %>%
  inner_join(trips,by=c("trip_id")) %>%
  inner_join(calendar,by=c("service_id")) %>%
  select(stop_id,stop_name,stop_lat,stop_lon) %>%
  group_by(stop_id,stop_name,stop_lat,stop_lon) %>%
  summarize(count=n()) %>%
  filter(count>=2000) # filter out least used stops

ggmap(lx_map, extent = "device") +
  geom_point(data = stops_freq,aes(x=stop_lon, y=stop_lat, size=count, fill=count), shape=21, alpha=0.8, colour = "blue")+ #plot stops with blue color
  scale_size_continuous(range = c(0, 2), guide = "none") + # size proportional to number of trips
  scale_fill_distiller()

