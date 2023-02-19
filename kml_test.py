import xml.dom.minidom
import sys


def createPlacemark(kmlDoc, row, order):
  # This creates a  element for a row of data.
  # A row is a dict.
  placemarkElement = kmlDoc.createElement('Placemark')
  extElement = kmlDoc.createElement('ExtendedData')
  placemarkElement.appendChild(extElement)
  
  # Loop through the columns and create a  element for every field that has a value.
  for key in order:
    if row[key]:
      dataElement = kmlDoc.createElement('Data')
      dataElement.setAttribute('name', key)
      valueElement = kmlDoc.createElement('value')
      dataElement.appendChild(valueElement)
      valueText = kmlDoc.createTextNode(row[key])
      valueElement.appendChild(valueText)
      extElement.appendChild(dataElement)
  
  pointElement = kmlDoc.createElement('Point')
  placemarkElement.appendChild(pointElement)
  coordinates = geocoding_for_kml.geocode(extractAddress(row))
  coorElement = kmlDoc.createElement('coordinates')
  coorElement.appendChild(kmlDoc.createTextNode(coordinates))
  pointElement.appendChild(coorElement)
  return placemarkElement

def createKML(fileName):
  # This constructs the KML document from the CSV file.
  kmlDoc = xml.dom.minidom.Document()
  
  kmlElement = kmlDoc.createElementNS('http://earth.google.com/kml/2.2', 'kml')
  kmlElement.setAttribute('xmlns','http://earth.google.com/kml/2.2')
  kmlElement = kmlDoc.appendChild(kmlElement)
  documentElement = kmlDoc.createElement('Document')
  documentElement = kmlElement.appendChild(documentElement)

  # Skip the header line.
  #csvReader.next()
  
  for row in range(2):
    placemarkElement = createPlacemark(kmlDoc, row, order)
    documentElement.appendChild(placemarkElement)
  kmlFile = open(fileName, 'w')
  kmlFile.write(kmlDoc.toprettyxml('  ', newl = '\n', encoding = 'utf-8'))

def main():
  # This reader opens up 'google-addresses.csv', which should be replaced with your own.
  # It creates a KML file called 'google.kml'.
  
  # If an argument was passed to the script, it splits the argument on a comma
  # and uses the resulting list to specify an order for when columns get added.
  # Otherwise, it defaults to the order used in the sample.
  
  #if len(sys.argv) >1: order = sys.argv[1].split(',')
  #else: order = ['Office','Address1','Address2','Address3','City','State','Zip','Phone','Fax']
  #csvreader = csv.DictReader(open('google-addresses.csv'),order)

  kml = createKML('google-addresses.kml')
if __name__ == '__main__':
  main()