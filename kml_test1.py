import xml.dom.minidom
import sys


def createLocs(locs):
  out = ''
  print("locs",locs[0][0])
  for i, loc in enumerate(locs):
      out += str(loc[0]) +','+ str(loc[1]) + ',0 '
  print(out)
  return out

def createKML(fileName):
  # This constructs the KML document from the CSV file.
  kmlDoc = xml.dom.minidom.Document()

  kmlElement = kmlDoc.createElementNS('http://earth.google.com/kml/2.2', 'kml')
  kmlElement.setAttribute('xmlns','http://earth.google.com/kml/2.2')
  kmlElement = kmlDoc.appendChild(kmlElement)

  documentElement = kmlDoc.createElement('Document')
  documentElement = kmlElement.appendChild(documentElement)
  name = kmlDoc.createElement('name')
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
  
  coordinates = createLocs([[22.13308794313373,52.423442704373],[22.13309765162534,52.42345109298137],[22.13310074044344,52.42345951214639],[22.1331171068907,52.4234707854348]])  
  coorElement = kmlDoc.createElement('coordinates')
  coorElement.appendChild(kmlDoc.createTextNode(coordinates))
  coorElement = LineString.appendChild(coorElement)
  #print("kmlDoc",kmlDoc)
  #print ("kmlDoc2",kmlDoc.toprettyxml(indent = '   '))

  with open(fileName, 'wb') as file:
    file.write(kmlDoc.toprettyxml('  ', newl = '\n', encoding = 'utf-8')  )

if __name__ == '__main__':
  kml = createKML("testP.kml")
