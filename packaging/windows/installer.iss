; SHADOW-6 Windows installer (Inno Setup 6).
; Built by packaging/windows/build.ps1, which passes the version and dist dir:
;   ISCC /DAppVersion=x.y.z /DDistDir=<repo>\dist\SHADOW-6 installer.iss
; Per-user install (no UAC prompt). User settings (%APPDATA%\SHADOW-6) and
; downloaded models are never touched by the uninstaller.

#define AppName "SHADOW-6"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef DistDir
  #define DistDir "..\..\dist\SHADOW-6"
#endif

[Setup]
AppId={{7487F8CA-4FA6-45E9-946B-344F9D91A4C7}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=SHADOW-6 Project
DefaultDirName={autopf}\{#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=SHADOW-6-Setup-{#AppVersion}-win-x64
SetupIconFile=..\..\assets\shadow6.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\SHADOW-6.exe
UninstallDisplayName={#AppName}
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked
Name: "autostart"; Description: "Start {#AppName} automatically when you sign in"; GroupDescription: "Startup:"

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\SHADOW-6.exe"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\SHADOW-6.exe"; Tasks: desktopicon

[Registry]
; Same Run-key value name the in-app "Start with Windows" toggle writes,
; so the installer checkbox and the app setting stay in sync.
Root: HKCU; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\SHADOW-6.exe"""; Tasks: autostart; Flags: uninsdeletevalue

[Run]
Filename: "{app}\SHADOW-6.exe"; Description: "Launch {#AppName} now"; Flags: nowait postinstall skipifsilent
