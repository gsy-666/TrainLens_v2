; TrainLens v0.1.0-beta Inno Setup Installer
; Private Python Runtime + Application Source

#define MyAppName "TrainLens"
#define MyAppVersion "0.1.0-beta"
#define MyAppPublisher "TrainLens"
#define MyAppURL "https://github.com/CVHub520/X-AnyLabeling"
#define MyAppExeName "TrainLens.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\Programs\TrainLens
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\..\dist\installer
OutputBaseFilename=TrainLens-v{#MyAppVersion}-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
UsePreviousAppDir=yes
DisableProgramGroupPage=auto
UninstallDisplayName={#MyAppName} {#MyAppVersion}
UninstallDisplayIcon={app}\resources\icon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: checkedonce

[Files]
; Python runtime
Source: "..\..\build\portable_runtime\runtime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
; Application source
Source: "..\..\anylabeling\*"; DestDir: "{app}\app\anylabeling"; Flags: ignoreversion recursesubdirs createallsubdirs
; Resources
Source: "..\..\anylabeling\resources\images\icon.ico"; DestDir: "{app}\resources"
; Configs
Source: "..\..\anylabeling\configs\*"; DestDir: "{app}\anylabeling\configs"; Flags: ignoreversion recursesubdirs createallsubdirs
; Auto-labeling configs
Source: "..\..\anylabeling\services\auto_labeling\configs\*"; DestDir: "{app}\anylabeling\services\auto_labeling\configs"; Flags: ignoreversion recursesubdirs createallsubdirs
; Path file to add app dir to sys.path
Source: "app.pth"; DestDir: "{app}\runtime\Lib\site-packages"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m anylabeling.app"; WorkingDir: "{app}\app"; IconFilename: "{app}\resources\icon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m anylabeling.app"; WorkingDir: "{app}\app"; IconFilename: "{app}\resources\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: "-m anylabeling.app"; WorkingDir: "{app}\app"; Description: "Launch TrainLens"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"

[Code]
// Keep user data on uninstall
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // User data is in %LOCALAPPDATA%\TrainLens — not in {app}
    // So it's automatically preserved.
  end;
end;
