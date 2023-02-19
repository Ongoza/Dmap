import xml.dom.minidom
import sys

def read_kml(filename):
    locs = []
    doc = xml.dom.minidom.parse(filename)  
    # doc.getElementsByTagName returns the NodeList
    tour = doc.getElementsByTagName("gx:Tour")[0]
    name = tour.getElementsByTagName("name")[0]
    print(name.firstChild.data)
    lst = doc.getElementsByTagName("gx:Playlist")[0]
    data = lst.getElementsByTagName("gx:FlyTo")    
    print(len(data))
    for loc in data:
        #longitude = float(loc.getElementsByTagName("longitude")[0].firstChild.data)
        #latitude = float(loc.getElementsByTagName("latitude")[0].firstChild.data)
        #print("longitude:% s, latitude:% s" % (longitude.firstChild.data, latitude.firstChild.data))
        locs.append([
            loc.getElementsByTagName("longitude")[0].firstChild.data,
            loc.getElementsByTagName("latitude")[0].firstChild.data
            ])
    print(locs)
    return locs

def createLocs(locs):
  out = ''
  print("locs",locs[0][0])
  for i, loc in enumerate(locs):
      out += str(loc[0]) +','+ str(loc[1]) + ',0 '
  print(out)
  return out

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
  
  nameP = kmlDoc.createElement('name')
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
  #locs = createLocs([[22.13308794313373,52.423442704373],[22.13309765162534,52.42345109298137],[22.13310074044344,52.42345951214639],[22.1331171068907,52.4234707854348]])  
  
  locs = createLocs(read_kml("Path3.kml"))
  kml = createKML("testP3.kml", locs)
