import xml.dom.minidom
import sys
import pyproj

def read_kml_line(filename):
    locs = []
    doc = xml.dom.minidom.parse(filename)  
    # doc.getElementsByTagName returns the NodeList
    doc_el = doc.getElementsByTagName("Document")[0]
    name = doc_el.getElementsByTagName("name")[0].firstChild.data
    print(name)
    place = doc_el.getElementsByTagName("Placemark")
    lst = place[0].getElementsByTagName("LineString")
    data = lst[0].getElementsByTagName("coordinates")[0].firstChild.data.split(' ')
    #print("3:", len(data))
    for loc in data:
        loc_a = loc.split(',')
        if len(loc_a)>1: locs.append([loc_a[0],loc_a[1]])
    #print(locs)
    #print(name)
    return name, locs

def createLocs(locs):
  out = ''
  print("locs",locs[0][0])
  for i, loc in enumerate(locs):
      out += str(loc[0]) +','+ str(loc[1]) + ',0 '
  print(out)
  return out

def createKML2(fileName, locs):
    # This constructs the KML document from the CSV file.
    kmlDoc = xml.dom.minidom.Document()
    kmlElement = kmlDoc.createElementNS('http://earth.google.com/kml/2.2', 'kml')
    kmlElement.setAttribute('xmlns','http://earth.google.com/kml/2.2')
    kmlElement = kmlDoc.appendChild(kmlElement)
    documentElement = kmlElement.appendChild(kmlDoc.createElement('Document'))
    name = documentElement.appendChild(kmlDoc.createElement('name'))
    name.appendChild(kmlDoc.createTextNode("TestP"))    
    Placemark = documentElement.appendChild(kmlDoc.createElement('Placemark'))    
    nameP = Placemark.appendChild(kmlDoc.createElement('name'))
    nameP.appendChild(kmlDoc.createTextNode("P11"))
    Style = Placemark.appendChild(kmlDoc.createElement('Style'))    
    LineStyle = Style.appendChild(kmlDoc.createElement('LineStyle'))
    color = LineStyle.appendChild(kmlDoc.createElement('color'))
    color.appendChild(kmlDoc.createTextNode("7fff0000"))
    width = LineStyle.appendChild(kmlDoc.createElement('width'))
    width.appendChild(kmlDoc.createTextNode("7"))    

    LineString = Placemark.appendChild(kmlDoc.createElement('LineString'))    
    tessellate = LineString.appendChild(kmlDoc.createElement('tessellate'))
    tessellate.appendChild(kmlDoc.createTextNode("1"))    
    coorElement = LineString.appendChild(kmlDoc.createElement('coordinates'))
    out_loc = ''
    for i, loc in enumerate(locs):
        out_loc += str(loc[0]) +','+ str(loc[1]) + ',0 '    
    coorElement.appendChild(kmlDoc.createTextNode(out_loc))
    with open(fileName, 'wb') as file:
        file.write(kmlDoc.toprettyxml('  ', newl = '\n', encoding = 'utf-8')  )

def createKML(fileName, locs):
  # This constructs the KML document from the CSV file.
  kmlDoc = xml.dom.minidom.Document()

  kmlElement = kmlDoc.createElementNS('http://earth.google.com/kml/2.2', 'kml')
  kmlElement.setAttribute('xmlns','http://earth.google.com/kml/2.2')
  kmlElement = kmlDoc.appendChild(kmlElement)

  documentElement = kmlDoc.createElement('Document')
  documentElement = kmlElement.appendChild(documentElement)
  name = kmlDoc.createElement('mame')
  name.appendChild(kmlDoc.createTextNode("TestP"))
  name = documentElement.appendChild(name)

  Placemark = kmlDoc.createElement('Placemark')
  Placemark = documentElement.appendChild(Placemark)
  
  nameP = kmlDoc.createElement('mame')
  nameP.appendChild(kmlDoc.createTextNode("P11"))
  nameP = Placemark.appendChild(nameP)

  LineString = kmlDoc.createElement('LineString')
  LineString = Placemark.appendChild(LineString)

  tessellate = kmlDoc.createElement('tessellate')
  tessellate.appendChild(kmlDoc.createTextNode("1"))
  tessellate = LineString.appendChild(tessellate)
  
  coorElement = kmlDoc.createElement('coordinates')
  coorElement.appendChild(kmlDoc.createTextNode(locs))
  coorElement = LineString.appendChild(coorElement)
  #print("kmlDoc",kmlDoc)
  #print ("kmlDoc2",kmlDoc.toprettyxml(indent = '   '))

  with open(fileName, 'wb') as file:
    file.write(kmlDoc.toprettyxml('  ', newl = '\n', encoding = 'utf-8')  )

if __name__ == '__main__':
  #locs = [[22.13308794313373,52.423442704373],[22.13309765162534,52.42345109298137],[22.13310074044344,52.42345951214639],[22.1331171068907,52.4234707854348]]
  name, gps_in = read_kml_line("testP22.kml")
  print(gps_in[0],name)
  wgs84_geod = pyproj.CRS('WGS 84').get_geod()
  az, _, dist = wgs84_geod.inv(gps_in[0][0], gps_in[0][1], gps_in[1][0], gps_in[1][1])
  print("delta", az, dist)
  lon, lat, _ = wgs84_geod.fwd(gps_in[0][0], gps_in[0][1], az, dist)
  print("lon, lat", lon, lat, gps_in[1])

 # createKML2("testP222.kml", locs)
