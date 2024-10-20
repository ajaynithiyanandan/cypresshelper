
import fs from 'fs';
import path from 'path';

const directoryPath = 'cypress-documentation/docs';

function getFileExtensions(dirPath, extensions = new Set()) {
const files = fs.readdirSync(dirPath);

files.forEach(file => {
  const filePath = path.join(dirPath, file);
  const stat = fs.statSync(filePath);

  if (stat.isDirectory()) {
    getFileExtensions(filePath, extensions);
  } else {
    const ext = path.extname(file);
    extensions.add(ext);
  }
});

return extensions;
}

const extensions = getFileExtensions(directoryPath);
console.log('File extensions:', Array.from(extensions));