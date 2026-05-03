; Maolan Generate Installer
; Run with: makensis.exe installer.nsi

!include "MUI2.nsh"
!include "LogicLib.nsh"

;--------------------------------
; General
;--------------------------------
Name "Maolan Generate"
OutFile "maolan-generate-setup.exe"
InstallDir "$LOCALAPPDATA\Maolan\bin"
InstallDirRegKey HKCU "Software\MaolanGenerate" "InstallDir"
RequestExecutionLevel user

;--------------------------------
; Version Info
;--------------------------------
VIProductVersion "0.0.4.0"
VIAddVersionKey "ProductName" "Maolan Generate"
VIAddVersionKey "ProductVersion" "0.0.4"
VIAddVersionKey "FileVersion" "0.0.4"
VIAddVersionKey "FileDescription" "Maolan AI Music Generation"
VIAddVersionKey "LegalCopyright" "BSD-2-Clause"

;--------------------------------
; Interface Settings
;--------------------------------
!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

;--------------------------------
; Pages
;--------------------------------
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

;--------------------------------
; Languages
;--------------------------------
!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Installer Sections
;--------------------------------
Section "Install"
    SetOutPath "$INSTDIR"

    ; Main binary
    File "C:\cargo-target\x86_64-pc-windows-msvc\release\maolan-generate.exe"

    ; VC++ Redistributable installer (bundled)
    File "..\vc_redist.x64.exe"
    ExecWait '"$INSTDIR\vc_redist.x64.exe" /install /quiet /norestart' $0
    Delete "$INSTDIR\vc_redist.x64.exe"

    ; Store installation folder
    WriteRegStr HKCU "Software\MaolanGenerate" "InstallDir" $INSTDIR

    ; Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Add to Add/Remove Programs
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "DisplayName" "Maolan Generate"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "DisplayVersion" "0.0.4"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "Publisher" "Maolan Team"
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "NoModify" 1
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate" \
        "NoRepair" 1

    ; Create Start Menu shortcut
    CreateDirectory "$SMPROGRAMS\Maolan Generate"
    CreateShortcut "$SMPROGRAMS\Maolan Generate\Maolan Generate.lnk" "$INSTDIR\maolan-generate.exe"
    CreateShortcut "$SMPROGRAMS\Maolan Generate\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
SectionEnd

;--------------------------------
; Uninstaller Section
;--------------------------------
Section "Uninstall"
    Delete "$INSTDIR\maolan-generate.exe"
    Delete "$INSTDIR\Uninstall.exe"

    Delete "$SMPROGRAMS\Maolan Generate\Maolan Generate.lnk"
    Delete "$SMPROGRAMS\Maolan Generate\Uninstall.lnk"
    RMDir "$SMPROGRAMS\Maolan Generate"

    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MaolanGenerate"
    DeleteRegKey HKCU "Software\MaolanGenerate"

    RMDir "$INSTDIR"
SectionEnd
